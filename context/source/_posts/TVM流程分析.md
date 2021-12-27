---
title: TVM流程分析
date: 2021-12-27 16:45:35
tags: 编译器 TVM
---

### 概述
TVM version：0.9.dev0
LLVM version: 10.0.0

本文主要用于跟踪relay.build的编译流程，生成机器码阶段.

### 整体流程
本文选取的例子代码位于： tvm/gallery/how_to/compile_models/from_tflite.py
流程如下：
    - 下载mobilenet_v1_1的模型文件,并解压.
    - 加载一张测试图片
    - 编译这个模型(relay.build)
    - 输入相关参数,然后运行模型(model.run)
    - 后处理,输出测试结果.

全文代码直接参考python文件
<!--more-->
### 主要代码流程
代码流程主要从relay.build开始
```
    ...
    lib = relay.build(mod, target, params=params)
    ...
```
此函数借口会调用build_module.py中的build函数
```
def build(
    ir_mod,
    target=None,
    target_host=None,
    executor=Executor("graph"),
    runtime=Runtime("cpp"),
    params=None,
    mod_name="default",
): 
    ...
     with tophub_context:
        bld_mod = BuildModule()
        graph_json, runtime_mod, params = bld_mod.build(
            mod=ir_mod,
            target=target,
            params=params,
            executor=executor,
            runtime=runtime,
            mod_name=mod_name,
        )
    ...
```
接下来跟进BuildModule()类
```
class BuildModule(object):
    """Build an IR module to run on TVM graph executor. This class is used
    to expose the `RelayBuildModule` APIs implemented in C++.
    """

    def __init__(self):
        self.mod = _build_module._BuildModule()
        self._get_graph_json = self.mod["get_graph_json"]
        self._get_module = self.mod["get_module"]
        self._build = self.mod["build"]
        self._optimize = self.mod["optimize"]
        self._set_params_func = self.mod["set_params"]
        self._get_params_func = self.mod["get_params"]
        self._get_function_metadata = self.mod["get_function_metadata"]
        self._get_devices = self.mod["get_devices"]
```
此时, self.mod["build"] 通过PackFunc进入C++代码层
具体实现在：tvm/src/relay/backend/build_module.cc
```
class RelayBuildModule : public runtime::ModuleNode {
 public:
  RelayBuildModule() = default;

  /*!
   * \brief Get member function to front-end
   * \param name The name of the function.
   * \param sptr_to_self The pointer to the module node.
   * \return The corresponding member function.
   */
  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) final {
    if (name == "get_graph_json") {
      return PackedFunc(
          [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->GetGraphJSON(); });
    } else if (name == "get_module") {
      return PackedFunc(
          [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->GetModule(); });
    } else if (name == "build") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        ICHECK_EQ(args.num_args, 6);
        this->Build(args[0], args[1], args[2], args[3], args[4], args[5]);
      });
    } else if (name == "list_params") {
      return PackedFunc(
          [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->ListParamNames(); });
    } else if (name == "get_params") {
        ...
```
然后继续跟进到this->Build函数中
```
void Build(IRModule mod, const TargetMap& targets, const tvm::Target& target_host,
             const Executor& executor, const Runtime& runtime, const String mod_name) {
    VLOG_CONTEXT << "Build";
    executor_ = executor;
    runtime_ = runtime;
    config_ = CompilationConfig(PassContext::Current(), targets, target_host);
    BuildRelay(std::move(mod), mod_name);
  }
```
跟进BuildRelay 函数
```
 void BuildRelay(IRModule relay_module, const String& mod_name) {
    // Relay IRModule -> IRModule optimizations.

    relay_module = OptimizeImpl(std::move(relay_module));

    // Get the updated function and new IRModule to build.
    // Instead of recreating the IRModule, we should look at the differences between this and the
    // incoming IRModule to see if we can just pass (IRModule, Function) to the code generator.
    Function func = Downcast<Function>(relay_module->Lookup("main"));

    IRModule func_module = WithAttrs(IRModule::FromExpr(func), {{tvm::attr::kExecutor, executor_},
                                                                {tvm::attr::kRuntime, runtime_}});
                                                    
    // Generate code for the updated function.
    executor_codegen_ = MakeExecutorCodegen(executor_->name);
    executor_codegen_->Init(nullptr, config_->legacy_target_map);
    executor_codegen_->Codegen(func_module, func, mod_name);
    executor_codegen_->UpdateOutput(&ret_);
    ret_.params = executor_codegen_->GetParams();

    auto lowered_funcs = executor_codegen_->GetIRModule();

    // No need to build for external functions.
    Target ext_dev("ext_dev");
    if (lowered_funcs.find(ext_dev) != lowered_funcs.end()) {
      lowered_funcs.Set(ext_dev, IRModule());
    }

    const Target& host_target = config_->host_se_scope->target;
    const runtime::PackedFunc* pf = runtime::Registry::Get("codegen.LLVMModuleCreate");

    // Generate a placeholder function that attaches linked params as its arguments.
    Bool should_link_params = func_module->ShouldLinkParameters();
    if (should_link_params) {
      CHECK(pf != nullptr) << "Unable to link-params with no target_host and no llvm codegen.";
      auto param_ids = executor_codegen_->GetParamIds();
      auto link_params = Map<String, tir::LinkedParam>();
      for (auto param : ret_.params) {
        link_params.Set(param.first, tir::LinkedParam(param_ids[param.first], param.second));
      }

      Map<String, ObjectRef> dict;
      dict.Set(tvm::tir::attr::kLinkedParams, link_params);
      dict.Set(tvm::attr::kGlobalSymbol, String(::tvm::runtime::symbol::tvm_lookup_linked_param));
      DictAttrs attrs{dict};
      auto prim = tir::PrimFunc(Array<tir::Var>(), tir::SeqStmt(Array<tir::Stmt>()), VoidType(),
                                Map<tir::Var, tir::Buffer>(), attrs);
      if (lowered_funcs.find(host_target) == lowered_funcs.end()) {
        lowered_funcs.Set(host_target,
                          IRModule(Map<GlobalVar, BaseFunc>({}), {}, {}, {}, func_module->attrs));
      }
      lowered_funcs[host_target]->Add(GlobalVar(::tvm::runtime::symbol::tvm_lookup_linked_param),
                                      prim);
    }

    // When there is no lowered_funcs due to reasons such as optimization.
    if (lowered_funcs.size() == 0) {
      if (host_target->kind->name == "llvm") {
        CHECK(pf != nullptr) << "Unable to create empty module for llvm without llvm codegen.";
        // If we can decide the target is LLVM, we then create an empty LLVM module.
        ret_.mod = (*pf)(host_target->str(), "empty_module");
      } else {
        // If we cannot decide the target is LLVM, we create an empty CSourceModule.
        // The code content is initialized with ";" to prevent complaining
        // from CSourceModuleNode::SaveToFile.
        ret_.mod = tvm::codegen::CSourceModuleCreate(";", "", Array<String>{});
      }
    } else {
      ret_.mod = tvm::build(lowered_funcs, host_target);
    }

    auto ext_mods = executor_codegen_->GetExternalModules();
    ret_.mod = tvm::codegen::CreateMetadataModule(ret_.params, ret_.mod, ext_mods, host_target,
                                                  runtime_, executor_codegen_->GetMetadata());

    // Remove external params which were stored in metadata module.
    for (tvm::runtime::Module mod : ext_mods) {
      auto pf_var = mod.GetFunction("get_const_vars");
      if (pf_var != nullptr) {
        Array<String> variables = pf_var();
        for (size_t i = 0; i < variables.size(); i++) {
          auto it = ret_.params.find(variables[i].operator std::string());
          if (it != ret_.params.end()) {
            ret_.params.erase(it);
          }
        }
      }
    }
  }
```
这是最关键的一段代码, 主要完成的工作如下：
- 1. OptimizeImpl函数主要是使用pass对relay IR进行优化
- 2. lowered_funcs 主要是低阶化,  即relay IR转化为 TVM IR.
- 3.  tvm::build: 调用tvm::build去调用后端代码生成器(这里是用LLVM)进行代码的生成.

首先看OptimizeImpl函数
```
IRModule OptimizeImpl(IRModule relay_module) {
    ICHECK(relay_module.defined()) << "The IRModule must be defined for the Relay compiler.";

    if (!params_.empty()) {
      ICHECK(relay_module->ContainGlobalVar("main")) << "Missing the main entry function";
      GlobalVar main_glb_var = relay_module->GetGlobalVar("main");
      Function main_func = Downcast<Function>(relay_module->Lookup(main_glb_var));
      auto new_main = BindParamsByName(main_func, params_);
      IRModuleNode* relay_module_ptr = relay_module.CopyOnWrite();
      relay_module_ptr->Update(main_glb_var, new_main);
    }

    Array<Pass> pass_seqs = GetPassPrefix(
        /*is_homogenous=*/config_->optional_homogeneous_target.defined(), /*is_vm=*/false);
    transform::PassContext pass_ctx = PassContext::Current();

    if (config_->optional_homogeneous_target.defined()) {
      // This pass currently only supports the homogeneous case.
      pass_seqs.push_back(transform::SplitArgs(
          config_->optional_homogeneous_target->GetAttr<Integer>("max_function_args", -1).value()));
    }
  
    // Always plan devices so the remaining passes don't need to distinguish homogeneous vs
    // hetrogenous execution.
    pass_seqs.push_back(transform::PlanDevices(config_));

    // Fuse the operations if it is needed.
    pass_seqs.push_back(transform::FuseOps());

    // Create a sequential pass and perform optimizations.
    transform::Pass seq = transform::Sequential(pass_seqs);

    if (config_->optional_homogeneous_target.defined()) {
      With<Target> tctx(config_->optional_homogeneous_target);
      relay_module = seq(relay_module);
    } else {
      relay_module = seq(relay_module);
    }

    // Do layout rewrite for auto-scheduler.
    if (backend::IsAutoSchedulerEnabled() && config_->optional_homogeneous_target.defined()) {
      Pass major_pass = transform::AutoSchedulerLayoutRewrite();
      bool enable_layout_rewrite_targets =
          config_->optional_homogeneous_target->kind->device_type == kDLCPU ||
          config_->optional_homogeneous_target->GetAttr<String>("device", "") == "mali";
      if (enable_layout_rewrite_targets && pass_ctx.PassEnabled(major_pass->Info())) {
        With<Target> tctx(config_->optional_homogeneous_target);
        relay_module = major_pass(relay_module);
        // Defuse ops to fold constants, then fuse them again
        relay_module = transform::DefuseOps()(relay_module);
        relay_module = transform::FoldConstant()(relay_module);
        relay_module = transform::FuseOps()(relay_module);
      }
    }

    relay_module = transform::InferType()(relay_module);

    // Inline the functions that have been lifted by the module scope.
    //
    // TODO(@zhiics) Note that we need to be careful about the subgraphs with
    // global function calls. We should make sure that these callees are also
    // inline functions. However, this should be very unlikely for accelerators
    // and vendor-provided libraries. So we don't handle for now.
    relay_module = transform::Inline()(relay_module);
    relay_module = transform::InferType()(relay_module);
    relay_module = transform::LabelOps()(relay_module);

    ICHECK(relay_module.defined());

    return relay_module;
  }
```
这里是用的pass的sequential结构, 即顺序执行添加到pass_seqs结构中的pass.
然后在relay_module = seq(relay_module); 去执行pass.
这里的seq是一个重载()运算符.
然后会调用transform.cc中的代码
```
IRModule Pass::operator()(IRModule mod) const {
  return this->operator()(std::move(mod), PassContext::Current());
}

IRModule Pass::operator()(IRModule mod, const PassContext& pass_ctx) const {
  const PassNode* node = operator->();
  ICHECK(node != nullptr);
  const PassInfo& pass_info = node->Info();
  if (!pass_ctx.InstrumentBeforePass(mod, pass_info)) {
    DLOG(INFO) << "Skipping pass : " << pass_info->name
               << " with opt level: " << pass_info->opt_level;
    return mod;
  }
  auto ret = node->operator()(std::move(mod), pass_ctx);
  pass_ctx.InstrumentAfterPass(ret, pass_info);
  return std::move(ret);
}
```
这一块的数据结构和细节后续再研究,
可以看到auto ret = node->operator()(std::move(mod), pass_ctx);
这行代码以后, Relay IR 的相关pass执行结束.

下面还有一些其他的相关pass执行以后,返回优化后的IR

接下来进行低阶化,做代码生成
```
    // Generate code for the updated function.
    executor_codegen_ = MakeExecutorCodegen(executor_->name);
    executor_codegen_->Init(nullptr, config_->legacy_target_map);
    executor_codegen_->Codegen(func_module, func, mod_name);
    executor_codegen_->UpdateOutput(&ret_);
    ret_.params = executor_codegen_->GetParams();

    auto lowered_funcs = executor_codegen_->GetIRModule();
```
executor_codegen_->Codegen的代码源码在tvm/src/relay/backend/graph_executor_codegen.cc中
```
 LoweredOutput Codegen(IRModule mod, relay::Function func, String mod_name) {
    mod_name_ = mod_name;
    VLOG_CONTEXT << "GraphExecutorCodegen";
    VLOG(1) << "compiling:" << std::endl << PrettyPrint(func);
    for (const auto& pair : targets_) {
      VLOG(1) << "target: " << pair.first << " = " << pair.second->ToDebugString();
    }

    // TODO(mbs): Why plan memory and update workspace sizes before lowering?
    memory_plan_ = GraphPlanMemory(func);

    backend::FunctionInfo func_info;

    if (memory_plan_.defined()) {
      // TODO(@electriclilies, @jroesch): remove UpdateMainWorkspaceSize
      // Switch from Map<Integer, Target> to undordered_map<DLDeviceType, Target> representation.
      // TODO(mbs): Plumb CompilationConfig through.
      tec::TargetMap tec_target_map;
      for (const auto& pair : targets_) {
        tec_target_map.emplace(static_cast<DLDeviceType>(pair.first->value), pair.second);
      }
      func_info = relay::tec::UpdateMainWorkspaceSize(mod, tec_target_map,
                                                      memory_plan_->expr_to_storage_info);
      mod = WithAttr(mod, "main_func_info", func_info);
    }

    // TODO(mbs): Plumb instead of reconstruct
    CompilationConfig config(transform::PassContext::Current(), targets_,
                             /*optional_host_target_arg=*/{});

    IRModule lowered_mod = tec::LowerTEPass(
        mod_name_,
        [this](BaseFunc func) {
          // We need to maintain the constant map for external
          // functions so we pass this processing function which
          // allows us to process each function as we lower it.
          if (func->GetAttr<String>(attr::kCompiler).defined()) {
            UpdateConstants(func, &params_);
          }

          // TODO(@areusch, @jroesch): We should refactor this to
          // execute as a further pass, instead writing data to the
          // lowering process directly.
          tec::UpdateFunctionMetadata(func, this->function_metadata_);
        },
        config->host_se_scope)(mod);

    Optional<backend::FunctionInfo> main_func_info =
        lowered_mod->GetAttr<backend::FunctionInfo>("main_func_info");

    function_metadata_.Set(runtime::symbol::tvm_module_main, main_func_info.value());

    Function lowered_main_func = Downcast<Function>(lowered_mod->Lookup("main"));

    // Now that we have lowered all operators to TIR code, we can proceed with compilation.
    //
    // We need to unfortunately re-plan as the previous results have been invalidated by lowering
    // we will fix this in future refactors.
    memory_plan_ = GraphPlanMemory(lowered_main_func);

    // The graph planner also can not handle planning calls to global variables to we must remap

    // First we convert all the parameters into input nodes.
    for (auto param : lowered_main_func->params) {
      auto node_ptr = GraphInputNode::make_node_ptr(param->name_hint(), GraphAttrs());
      var_map_[param.get()] = AddNode(node_ptr, param);
    }

    heads_ = VisitExpr(lowered_main_func->body);
    std::ostringstream os;

    dmlc::JSONWriter writer(&os);
    GetJSON(&writer);
    LoweredOutput ret;
    ret.graph_json = os.str();
    ret.params = std::unordered_map<std::string, std::pair<int, const tvm::runtime::NDArray>>();
    for (auto param : params_) {
      ret.params.emplace(std::make_pair(
          param.first,
          std::make_pair(static_cast<int>(param_storage_ids_[param.first]), param.second)));
    }
    ret.function_metadata = std::move(function_metadata_);

    Optional<Array<tvm::runtime::Module>> external_modules =
        lowered_mod->GetAttr<Array<tvm::runtime::Module>>("external_mods");
    ICHECK(external_modules) << "Attribute \"external_mods\" should be set at this point.";

    // This is the point where we separate the functions in the module by target
    ret.lowered_funcs = tec::GetPerTargetModules(lowered_mod);
    ret.external_mods = external_modules.value();
    return ret;
  }
```
此时的lowered_funcs就是低阶化的IR, 即tir.

接下来调用tvm::build来根据后端生成汇编代码
调用此函数会进入tvm/src/driver/driver_api.cc
```
runtime::Module build(const Map<Target, IRModule>& inputs_arg, const Target& target_host_arg) {
  auto pass_ctx = transform::PassContext::Current();

  std::vector<runtime::Module> device_modules;
  Map<Target, IRModule> inputs = inputs_arg;
  Target target_host = target_host_arg;

  // Fetch previous defined target host in targets
  CheckAndUpdateHostConsistency(&inputs, &target_host);

  if (!target_host.defined()) {
    for (const auto& it : inputs) {
      if (it.first->kind->device_type == kDLCPU || it.first->kind->device_type == kDLMicroDev) {
        target_host = it.first;
        break;
      }
    }
  }

  if (!target_host.defined()) {
    target_host = DefaultTargetHost(target_host);
  }

  // Update target host for all targets
  CheckAndUpdateHostConsistency(&inputs, &target_host);

  // Take the attrs from the first module so the eventual modules have them.
  // Ideally this would just be one unified module all the way through;
  IRModule first_module = (*inputs.begin()).second;
  IRModule mhost_all = IRModule(Map<GlobalVar, BaseFunc>(), {}, {}, {}, first_module->attrs);

  ICHECK(mhost_all.defined()) << "The host module must be defined";

  for (const auto& it : inputs) {
    if (it.second.defined()) {
      const Target& target = it.first;
      const IRModule& ir_module = it.second;
      auto pair = SplitMixedModule(ir_module, target, target_host);
      auto& host_mod = pair.first;
      auto& device_mod = pair.second;

      ICHECK(host_mod.defined()) << "The split host module must be defined";

      ICHECK(mhost_all.defined()) << "The host module must be defined";

      // We don't want library modules going back into host codegen
      // unless they're supposed to. Here if we overrode the target host
      // to allow lowering previously we check that it's meant to be placed
      // back into the host Module.
      bool overrides_host_target = target->kind->device_type == target_host->kind->device_type;
      bool non_host_target_kind = target->kind != target_host->kind;
      if (overrides_host_target && non_host_target_kind) {
        device_modules.push_back(codegen::Build(host_mod, it.first));
      } else {
        mhost_all->Update(host_mod);
      }

      if (device_mod->functions.size() != 0) {
        device_modules.push_back(codegen::Build(device_mod, it.first));
      }
    }
  }

  runtime::Module mhost = codegen::Build(mhost_all, target_host);

  for (const auto& it : device_modules) {
    if (it.operator->()) {
      mhost.Import(it);
    }
  }

  return mhost;
}
```

这段代码最核心的是codegen::Build
跳转过来就是代码生成模块：tvm/src/target/codegen.cc
```
runtime::Module Build(IRModule mod, Target target) {
  if (transform::PassContext::Current()
          ->GetConfig<Bool>("tir.disable_assert", Bool(false))
          .value()) {
    mod = tir::transform::SkipAssert()(mod);
  }

  auto target_attr_map = tvm::TargetKind::GetAttrMap<FTVMTIRToRuntime>("TIRToRuntime");
  if (target_attr_map.count(target->kind)) {
    return target_attr_map[target->kind](mod, target);
  }

  // the build function.
  std::string build_f_name = "target.build." + target->kind->name;
  const PackedFunc* bf = runtime::Registry::Get(build_f_name);
  ICHECK(bf != nullptr) << build_f_name << " is not enabled";
  return (*bf)(mod, target);
}
```
我们目前的后端是LLVM,所以target.build.llvm会跳转到tvm/src/target/llvm/llvm_module.cc中
```
TVM_REGISTER_GLOBAL("target.build.llvm")
    .set_body_typed([](IRModule mod, Target target) -> runtime::Module {
      auto n = make_object<LLVMModuleNode>();
      n->Init(mod, target);
      return runtime::Module(n);
    });
```
直接调用当前文件中的Init函数
```
void Init(const IRModule& mod, const Target& target) {
    InitializeLLVM();
    tm_ = GetLLVMTargetMachine(target);
    ctx_ = std::make_shared<llvm::LLVMContext>();
    std::unique_ptr<CodeGenLLVM> cg = CodeGenLLVM::Create(tm_.get());

    std::vector<PrimFunc> funcs;
    std::string entry_func;
    Map<String, LinkedParam> linked_params;
    bool found_linked_params = false;
    bool could_have_linked_params = mod->ShouldLinkParameters();
    relay::Runtime runtime =
        mod->GetAttr<relay::Runtime>(tvm::attr::kRuntime).value_or(relay::Runtime::Create("cpp"));
    bool system_lib = runtime->GetAttr<Bool>("system-lib").value_or(Bool(false));
    bool target_c_runtime = runtime->name == "crt";

    for (auto kv : mod->functions) {
      if (could_have_linked_params &&
          kv.first->name_hint == ::tvm::runtime::symbol::tvm_lookup_linked_param) {
        Map<String, ObjectRef> attrs_dict =
            Downcast<Map<String, ObjectRef>>(kv.second->attrs->dict);
        CHECK(attrs_dict.find(::tvm::tir::attr::kLinkedParams) != attrs_dict.end())
            << "no " << ::tvm::tir::attr::kLinkedParams << " attribute found!";
        linked_params =
            Downcast<Map<String, LinkedParam>>(attrs_dict[::tvm::tir::attr::kLinkedParams]);
        found_linked_params = true;
        continue;
      }
      if (!kv.second->IsInstance<PrimFuncNode>()) {
        // (@jroesch): we relax constraints here, Relay functions will just be ignored.
        DLOG(INFO) << "Can only lower IR Module with PrimFuncs, but got "
                   << kv.second->GetTypeKey();
        continue;
      }
      auto f = Downcast<PrimFunc>(kv.second);
      auto global_symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol);
      ICHECK(global_symbol.defined());
      function_names_.push_back(global_symbol.value());
      if (f->HasNonzeroAttr(tir::attr::kIsEntryFunc)) {
        entry_func = global_symbol.value();
      }
      funcs.push_back(f);
    }

    // TODO(@jroesch): follow up on this condition.
    // ICHECK(funcs.size() > 0 || (could_have_linked_params && found_linked_params));
    // TODO(tqchen): remove the entry function behavior as it does not
    // makes sense when we start to use multiple modules.
    cg->Init("TVMMod", tm_.get(), ctx_.get(), system_lib, system_lib, target_c_runtime);

    // See https://llvm.org/docs/LangRef.html#fast-math-flags for details
    Bool fast_math_all = target->GetAttr<Bool>("fast-math").value_or(Bool(false));
    Bool fast_math_nnan = target->GetAttr<Bool>("fast-math-nnan").value_or(Bool(false));
    Bool fast_math_ninf = target->GetAttr<Bool>("fast-math-ninf").value_or(Bool(false));
    Bool fast_math_nsz = target->GetAttr<Bool>("fast-math-nsz").value_or(Bool(false));
    Bool fast_math_arcp = target->GetAttr<Bool>("fast-math-arcp").value_or(Bool(false));

    llvm::FastMathFlags fmf;
    if (fast_math_all) {
#if TVM_LLVM_VERSION >= 60
      fmf.setFast();
#else
      fmf.setUnsafeAlgebra();
#endif
    }

    if (fast_math_nnan) {
      fmf.setNoNaNs();
    }
    if (fast_math_ninf) {
      fmf.setNoInfs();
    }
    if (fast_math_nsz) {
      fmf.setNoSignedZeros();
    }
    if (fast_math_arcp) {
      fmf.setAllowReciprocal();
    }

#if TVM_LLVM_VERSION >= 60
    Bool fast_math_contract = target->GetAttr<Bool>("fast-math-contract").value_or(Bool(false));
    Bool fast_math_afn = target->GetAttr<Bool>("fast-math-afn").value_or(Bool(false));
    Bool fast_math_reassoc = target->GetAttr<Bool>("fast-math-reassoc").value_or(Bool(false));
    if (fast_math_contract) {
      fmf.setAllowContract(true);
    }
    if (fast_math_afn) {
      fmf.setApproxFunc();
    }
    if (fast_math_reassoc) {
      fmf.setAllowReassoc();
    }
#endif

    cg->SetFastMathFlag(fmf);

    cg->AddFunctionsOrdered(funcs.begin(), funcs.end());

    if (entry_func.length() != 0) {
      cg->AddMainFunction(entry_func);
    }

    if (found_linked_params) {
      cg->LinkParameters(linked_params);
    }
    module_ = cg->Finish();

    module_->addModuleFlag(llvm::Module::Warning, "tvm_target",
                           llvm::MDString::get(*ctx_, LLVMTargetToString(target)));
    module_->addModuleFlag(llvm::Module::Override, "Debug Info Version",
                           llvm::DEBUG_METADATA_VERSION);

    if (tm_->getTargetTriple().isOSDarwin()) {
      module_->addModuleFlag(llvm::Module::Override, "Dwarf Version", 2);
    }

    std::string verify_errors_storage;
    llvm::raw_string_ostream verify_errors(verify_errors_storage);
    LOG_IF(FATAL, llvm::verifyModule(*module_, &verify_errors))
        << "LLVM module verification failed with the following errors: \n"
        << verify_errors.str();
    target_ = target;
    mptr_ = module_.get();

    module_->print(llvm::errs(), nullptr);
  }
```
这段代码会对目前相关进行优化.
我们这里可以添加一行代码module_->print(llvm::errs(), nullptr);
可以将生成的汇编代码打印出来.

### IR
调试过程中的一些中间文件保存在github目录中：
https://github.com/ah-cheng/TvmIrLog

