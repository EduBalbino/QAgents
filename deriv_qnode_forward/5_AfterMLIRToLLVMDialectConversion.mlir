module @deriv_qnode_forward {
  llvm.func @__catalyst__rt__finalize()
  llvm.func @__catalyst__rt__initialize(!llvm.ptr)
  llvm.func @__catalyst__rt__device_release()
  llvm.func @__catalyst__rt__qubit_release_array(!llvm.ptr)
  llvm.func @__catalyst__qis__Expval(i64) -> f64
  llvm.func @__catalyst__qis__NamedObs(i64, !llvm.ptr) -> i64
  llvm.func @__catalyst__qis__CNOT(!llvm.ptr, !llvm.ptr, !llvm.ptr)
  llvm.func @__catalyst__qis__RZ(f64, !llvm.ptr, !llvm.ptr)
  llvm.func @__catalyst__qis__RY(f64, !llvm.ptr, !llvm.ptr)
  llvm.func @__catalyst__rt__array_get_element_ptr_1d(!llvm.ptr, i64) -> !llvm.ptr
  llvm.func @__catalyst__rt__qubit_allocate_array(i64) -> !llvm.ptr
  llvm.mlir.global internal constant @"{}"("{}\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @LightningGPUSimulator("LightningGPUSimulator\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @"/home/pichau/QAgents/.venv/lib/python3.12/site-packages/pennylane_lightning/liblightning_gpu_catalyst.so"("/home/pichau/QAgents/.venv/lib/python3.12/site-packages/pennylane_lightning/liblightning_gpu_catalyst.so\00") {addr_space = 0 : i32}
  llvm.func @__catalyst__rt__device_init(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i1)
  llvm.func @__catalyst__qis__Gradient(i64, ...)
  llvm.func @__catalyst__rt__toggle_recorder(i1)
  llvm.mlir.global linkonce constant @enzyme_dupnoneed(0 : i8) {addr_space = 0 : i32} : i8
  llvm.mlir.global linkonce constant @enzyme_const(0 : i8) {addr_space = 0 : i32} : i8
  llvm.func @__enzyme_autodiff0(...)
  llvm.mlir.global external @__enzyme_function_like_free() {addr_space = 0 : i32} : !llvm.array<2 x ptr> {
    %0 = llvm.mlir.addressof @_mlir_memref_to_llvm_free : !llvm.ptr
    %1 = llvm.mlir.addressof @freename : !llvm.ptr
    %2 = llvm.mlir.undef : !llvm.array<2 x ptr>
    %3 = llvm.insertvalue %0, %2[0] : !llvm.array<2 x ptr> 
    %4 = llvm.insertvalue %1, %3[1] : !llvm.array<2 x ptr> 
    llvm.return %4 : !llvm.array<2 x ptr>
  }
  llvm.mlir.global linkonce constant @freename("free\00") {addr_space = 0 : i32}
  llvm.mlir.global linkonce constant @dealloc_indices("-1\00") {addr_space = 0 : i32}
  llvm.mlir.global external @__enzyme_allocation_like() {addr_space = 0 : i32} : !llvm.array<4 x ptr> {
    %0 = llvm.mlir.undef : !llvm.array<4 x ptr>
    %1 = llvm.mlir.addressof @_mlir_memref_to_llvm_free : !llvm.ptr
    %2 = llvm.mlir.addressof @dealloc_indices : !llvm.ptr
    %3 = llvm.mlir.addressof @_mlir_memref_to_llvm_alloc : !llvm.ptr
    %4 = llvm.mlir.constant(0 : index) : i64
    %5 = llvm.inttoptr %4 : i64 to !llvm.ptr
    %6 = llvm.insertvalue %3, %0[0] : !llvm.array<4 x ptr> 
    %7 = llvm.insertvalue %5, %6[1] : !llvm.array<4 x ptr> 
    %8 = llvm.insertvalue %2, %7[2] : !llvm.array<4 x ptr> 
    %9 = llvm.insertvalue %1, %8[3] : !llvm.array<4 x ptr> 
    llvm.return %9 : !llvm.array<4 x ptr>
  }
  llvm.func @_mlir_memref_to_llvm_free(!llvm.ptr)
  llvm.func @_mlir_memref_to_llvm_alloc(i64) -> !llvm.ptr
  llvm.mlir.global external @__enzyme_register_gradient_qnode_forward_0.quantum() {addr_space = 0 : i32} : !llvm.array<3 x ptr> {
    %0 = llvm.mlir.undef : !llvm.array<3 x ptr>
    %1 = llvm.mlir.addressof @qnode_forward_0.quantum.customqgrad : !llvm.ptr
    %2 = llvm.mlir.addressof @qnode_forward_0.quantum.augfwd : !llvm.ptr
    %3 = llvm.mlir.addressof @qnode_forward_0.quantum : !llvm.ptr
    %4 = llvm.insertvalue %3, %0[0] : !llvm.array<3 x ptr> 
    %5 = llvm.insertvalue %2, %4[1] : !llvm.array<3 x ptr> 
    %6 = llvm.insertvalue %1, %5[2] : !llvm.array<3 x ptr> 
    llvm.return %6 : !llvm.array<3 x ptr>
  }
  llvm.mlir.global private constant @__constant_xf32(3.14159274 : f32) {addr_space = 0 : i32, alignment = 64 : i64} : f32
  llvm.func @jit_deriv_qnode_forward(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: !llvm.ptr, %arg10: !llvm.ptr, %arg11: i64, %arg12: i64, %arg13: i64) -> !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> attributes {llvm.copy_memref, llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(24 : index) : i64
    %1 = llvm.mlir.constant(3 : index) : i64
    %2 = llvm.mlir.constant(8 : index) : i64
    %3 = llvm.mlir.constant(0 : index) : i64
    %4 = llvm.mlir.constant(64 : index) : i64
    %5 = llvm.mlir.zero : !llvm.ptr
    %6 = llvm.mlir.constant(1 : index) : i64
    %7 = llvm.mlir.constant(0.000000e+00 : f64) : f64
    %8 = llvm.mlir.constant(1.000000e+00 : f64) : f64
    %9 = llvm.mlir.constant(3735928559 : index) : i64
    %10 = llvm.mlir.addressof @qnode_forward_0.preprocess : !llvm.ptr
    %11 = llvm.mlir.addressof @enzyme_const : !llvm.ptr
    %12 = llvm.mlir.addressof @enzyme_dupnoneed : !llvm.ptr
    %13 = llvm.mlir.constant(4 : index) : i64
    %14 = llvm.mlir.constant(0 : i8) : i8
    %15 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %16 = llvm.call @qnode_forward_0.pcount(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> i64
    %17 = llvm.getelementptr %5[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %18 = llvm.ptrtoint %17 : !llvm.ptr to i64
    %19 = llvm.add %18, %4 : i64
    %20 = llvm.call @_mlir_memref_to_llvm_alloc(%19) : (i64) -> !llvm.ptr
    %21 = llvm.ptrtoint %20 : !llvm.ptr to i64
    %22 = llvm.sub %4, %6 : i64
    %23 = llvm.add %21, %22 : i64
    %24 = llvm.urem %23, %4 : i64
    %25 = llvm.sub %23, %24 : i64
    %26 = llvm.inttoptr %25 : i64 to !llvm.ptr
    llvm.store %7, %26 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    llvm.store %8, %26 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %27 = llvm.getelementptr %5[96] : (!llvm.ptr) -> !llvm.ptr, f32
    %28 = llvm.ptrtoint %27 : !llvm.ptr to i64
    %29 = llvm.call @_mlir_memref_to_llvm_alloc(%28) : (i64) -> !llvm.ptr
    %30 = llvm.insertvalue %29, %15[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %31 = llvm.insertvalue %29, %30[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %32 = llvm.insertvalue %3, %31[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %33 = llvm.insertvalue %13, %32[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %34 = llvm.insertvalue %2, %33[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %35 = llvm.insertvalue %1, %34[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %36 = llvm.insertvalue %0, %35[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %37 = llvm.insertvalue %1, %36[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %38 = llvm.insertvalue %6, %37[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %39 = llvm.getelementptr %5[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %40 = llvm.ptrtoint %39 : !llvm.ptr to i64
    %41 = llvm.call @_mlir_memref_to_llvm_alloc(%40) : (i64) -> !llvm.ptr
    %42 = llvm.mul %13, %0 : i64
    %43 = llvm.add %42, %3 : i64
    %44 = llvm.mul %43, %13 : i64
    "llvm.intr.memset"(%29, %14, %44) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
    llvm.call @__enzyme_autodiff0(%10, %11, %arg0, %arg1, %29, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %11, %arg9, %11, %arg10, %arg11, %arg12, %arg13, %11, %16, %11, %41, %12, %41, %26, %3) vararg(!llvm.func<void (...)>) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%41) : (!llvm.ptr) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%20) : (!llvm.ptr) -> ()
    %45 = llvm.ptrtoint %29 : !llvm.ptr to i64
    %46 = llvm.icmp "eq" %9, %45 : i64
    llvm.cond_br %46, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %47 = llvm.getelementptr %5[96] : (!llvm.ptr) -> !llvm.ptr, f32
    %48 = llvm.ptrtoint %47 : !llvm.ptr to i64
    %49 = llvm.call @_mlir_memref_to_llvm_alloc(%48) : (i64) -> !llvm.ptr
    %50 = llvm.insertvalue %49, %15[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %51 = llvm.insertvalue %49, %50[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %52 = llvm.insertvalue %3, %51[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %53 = llvm.insertvalue %13, %52[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %54 = llvm.insertvalue %2, %53[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %55 = llvm.insertvalue %1, %54[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %56 = llvm.insertvalue %0, %55[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %57 = llvm.insertvalue %1, %56[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %58 = llvm.insertvalue %6, %57[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %59 = llvm.mul %13, %6 : i64
    %60 = llvm.mul %59, %2 : i64
    %61 = llvm.mul %60, %1 : i64
    %62 = llvm.getelementptr %5[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %63 = llvm.ptrtoint %62 : !llvm.ptr to i64
    %64 = llvm.mul %61, %63 : i64
    "llvm.intr.memcpy"(%49, %29, %64) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.br ^bb3(%58 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>)
  ^bb2:  // pred: ^bb0
    llvm.br ^bb3(%38 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>)
  ^bb3(%65: !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>):  // 2 preds: ^bb1, ^bb2
    llvm.br ^bb4
  ^bb4:  // pred: ^bb3
    llvm.return %65 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
  }
  llvm.func @_catalyst_pyface_jit_deriv_qnode_forward(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
    %0 = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<(ptr, ptr, ptr)>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr, ptr, ptr)> 
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, ptr)> 
    llvm.call @_catalyst_ciface_jit_deriv_qnode_forward(%arg0, %1, %2) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }
  llvm.func @_catalyst_ciface_jit_deriv_qnode_forward(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) attributes {llvm.copy_memref, llvm.emit_c_interface} {
    %0 = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %5 = llvm.extractvalue %0[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %6 = llvm.extractvalue %0[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %7 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %8 = llvm.extractvalue %0[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %9 = llvm.extractvalue %0[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %10 = llvm.load %arg2 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %11 = llvm.extractvalue %10[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %12 = llvm.extractvalue %10[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %13 = llvm.extractvalue %10[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %14 = llvm.extractvalue %10[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %15 = llvm.extractvalue %10[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %16 = llvm.call @jit_deriv_qnode_forward(%1, %2, %3, %4, %5, %6, %7, %8, %9, %11, %12, %13, %14, %15) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    llvm.store %16, %arg0 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>, !llvm.ptr
    llvm.return
  }
  llvm.func internal @qnode_forward_0(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: !llvm.ptr, %arg10: !llvm.ptr, %arg11: i64, %arg12: i64, %arg13: i64) -> !llvm.struct<(ptr, ptr, i64)> attributes {diff_method = "adjoint", qnode} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.mlir.constant(2 : i64) : i64
    %2 = llvm.mlir.constant(3 : i64) : i64
    %3 = llvm.mlir.constant(4 : i64) : i64
    %4 = llvm.mlir.constant(5 : i64) : i64
    %5 = llvm.mlir.constant(6 : i64) : i64
    %6 = llvm.mlir.constant(7 : i64) : i64
    %7 = llvm.mlir.constant(8 : i64) : i64
    %8 = llvm.mlir.constant(false) : i1
    %9 = llvm.mlir.addressof @"{}" : !llvm.ptr
    %10 = llvm.mlir.addressof @LightningGPUSimulator : !llvm.ptr
    %11 = llvm.mlir.addressof @"/home/pichau/QAgents/.venv/lib/python3.12/site-packages/pennylane_lightning/liblightning_gpu_catalyst.so" : !llvm.ptr
    %12 = llvm.mlir.constant(64 : index) : i64
    %13 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64)>
    %14 = llvm.mlir.addressof @__constant_xf32 : !llvm.ptr
    %15 = llvm.mlir.zero : !llvm.ptr
    %16 = llvm.mlir.constant(0 : i64) : i64
    %17 = llvm.mlir.constant(0 : index) : i64
    %18 = llvm.mlir.constant(8 : index) : i64
    %19 = llvm.mlir.constant(1 : index) : i64
    %20 = llvm.getelementptr inbounds %arg1[74] : (!llvm.ptr) -> !llvm.ptr, f32
    %21 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %22 = llvm.getelementptr inbounds %arg1[73] : (!llvm.ptr) -> !llvm.ptr, f32
    %23 = llvm.load %22 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %24 = llvm.getelementptr inbounds %arg1[72] : (!llvm.ptr) -> !llvm.ptr, f32
    %25 = llvm.load %24 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %26 = llvm.getelementptr inbounds %arg1[50] : (!llvm.ptr) -> !llvm.ptr, f32
    %27 = llvm.load %26 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %28 = llvm.getelementptr inbounds %arg1[49] : (!llvm.ptr) -> !llvm.ptr, f32
    %29 = llvm.load %28 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %30 = llvm.getelementptr inbounds %arg1[48] : (!llvm.ptr) -> !llvm.ptr, f32
    %31 = llvm.load %30 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %32 = llvm.getelementptr inbounds %arg1[44] : (!llvm.ptr) -> !llvm.ptr, f32
    %33 = llvm.load %32 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %34 = llvm.getelementptr inbounds %arg1[43] : (!llvm.ptr) -> !llvm.ptr, f32
    %35 = llvm.load %34 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %36 = llvm.getelementptr inbounds %arg1[42] : (!llvm.ptr) -> !llvm.ptr, f32
    %37 = llvm.load %36 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %38 = llvm.getelementptr inbounds %arg1[23] : (!llvm.ptr) -> !llvm.ptr, f32
    %39 = llvm.load %38 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %40 = llvm.getelementptr inbounds %arg1[22] : (!llvm.ptr) -> !llvm.ptr, f32
    %41 = llvm.load %40 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %42 = llvm.getelementptr inbounds %arg1[21] : (!llvm.ptr) -> !llvm.ptr, f32
    %43 = llvm.load %42 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %44 = llvm.getelementptr %15[8] : (!llvm.ptr) -> !llvm.ptr, f32
    %45 = llvm.ptrtoint %44 : !llvm.ptr to i64
    %46 = llvm.add %45, %12 : i64
    %47 = llvm.call @_mlir_memref_to_llvm_alloc(%46) : (i64) -> !llvm.ptr
    %48 = llvm.ptrtoint %47 : !llvm.ptr to i64
    %49 = llvm.sub %12, %19 : i64
    %50 = llvm.add %48, %49 : i64
    %51 = llvm.urem %50, %12 : i64
    %52 = llvm.sub %50, %51 : i64
    %53 = llvm.inttoptr %52 : i64 to !llvm.ptr
    llvm.br ^bb1(%17 : i64)
  ^bb1(%54: i64):  // 2 preds: ^bb0, ^bb2
    %55 = llvm.icmp "slt" %54, %18 : i64
    llvm.cond_br %55, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %56 = llvm.load %14 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %57 = llvm.getelementptr inbounds %53[%54] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %56, %57 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f32, !llvm.ptr
    %58 = llvm.add %54, %19 : i64
    llvm.br ^bb1(%58 : i64)
  ^bb3:  // pred: ^bb1
    llvm.br ^bb4(%17 : i64)
  ^bb4(%59: i64):  // 2 preds: ^bb3, ^bb5
    %60 = llvm.icmp "slt" %59, %18 : i64
    llvm.cond_br %60, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %61 = llvm.getelementptr inbounds %53[%59] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %62 = llvm.load %61 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %63 = llvm.getelementptr inbounds %arg10[%59] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %64 = llvm.load %63 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %65 = llvm.fmul %62, %64 : f32
    %66 = llvm.getelementptr inbounds %53[%59] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %65, %66 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f32, !llvm.ptr
    %67 = llvm.add %59, %19 : i64
    llvm.br ^bb4(%67 : i64)
  ^bb6:  // pred: ^bb4
    %68 = llvm.getelementptr inbounds %53[7] : (!llvm.ptr) -> !llvm.ptr, f32
    %69 = llvm.load %68 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %70 = llvm.getelementptr inbounds %11[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<105 x i8>
    %71 = llvm.getelementptr inbounds %10[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<22 x i8>
    %72 = llvm.getelementptr inbounds %9[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3 x i8>
    llvm.call @__catalyst__rt__device_init(%70, %71, %72, %16, %8) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i1) -> ()
    %73 = llvm.call @__catalyst__rt__qubit_allocate_array(%7) : (i64) -> !llvm.ptr
    %74 = llvm.call @__catalyst__rt__array_get_element_ptr_1d(%73, %6) : (!llvm.ptr, i64) -> !llvm.ptr
    %75 = llvm.load %74 : !llvm.ptr -> !llvm.ptr
    %76 = llvm.fpext %69 : f32 to f64
    llvm.call @__catalyst__qis__RY(%76, %75, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %77 = llvm.fpext %43 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%77, %75, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %78 = llvm.fpext %41 : f32 to f64
    llvm.call @__catalyst__qis__RY(%78, %75, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %79 = llvm.fpext %39 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%79, %75, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %80 = llvm.getelementptr inbounds %arg1[20] : (!llvm.ptr) -> !llvm.ptr, f32
    %81 = llvm.load %80 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %82 = llvm.getelementptr inbounds %arg1[19] : (!llvm.ptr) -> !llvm.ptr, f32
    %83 = llvm.load %82 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %84 = llvm.getelementptr inbounds %arg1[18] : (!llvm.ptr) -> !llvm.ptr, f32
    %85 = llvm.load %84 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %86 = llvm.getelementptr inbounds %53[6] : (!llvm.ptr) -> !llvm.ptr, f32
    %87 = llvm.load %86 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %88 = llvm.call @__catalyst__rt__array_get_element_ptr_1d(%73, %5) : (!llvm.ptr, i64) -> !llvm.ptr
    %89 = llvm.load %88 : !llvm.ptr -> !llvm.ptr
    %90 = llvm.fpext %87 : f32 to f64
    llvm.call @__catalyst__qis__RY(%90, %89, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %91 = llvm.fpext %85 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%91, %89, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %92 = llvm.fpext %83 : f32 to f64
    llvm.call @__catalyst__qis__RY(%92, %89, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %93 = llvm.fpext %81 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%93, %89, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %94 = llvm.getelementptr inbounds %arg1[17] : (!llvm.ptr) -> !llvm.ptr, f32
    %95 = llvm.load %94 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %96 = llvm.getelementptr inbounds %arg1[16] : (!llvm.ptr) -> !llvm.ptr, f32
    %97 = llvm.load %96 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %98 = llvm.getelementptr inbounds %arg1[15] : (!llvm.ptr) -> !llvm.ptr, f32
    %99 = llvm.load %98 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %100 = llvm.getelementptr inbounds %53[5] : (!llvm.ptr) -> !llvm.ptr, f32
    %101 = llvm.load %100 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %102 = llvm.call @__catalyst__rt__array_get_element_ptr_1d(%73, %4) : (!llvm.ptr, i64) -> !llvm.ptr
    %103 = llvm.load %102 : !llvm.ptr -> !llvm.ptr
    %104 = llvm.fpext %101 : f32 to f64
    llvm.call @__catalyst__qis__RY(%104, %103, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %105 = llvm.fpext %99 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%105, %103, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %106 = llvm.fpext %97 : f32 to f64
    llvm.call @__catalyst__qis__RY(%106, %103, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %107 = llvm.fpext %95 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%107, %103, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %108 = llvm.getelementptr inbounds %arg1[14] : (!llvm.ptr) -> !llvm.ptr, f32
    %109 = llvm.load %108 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %110 = llvm.getelementptr inbounds %arg1[13] : (!llvm.ptr) -> !llvm.ptr, f32
    %111 = llvm.load %110 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %112 = llvm.getelementptr inbounds %arg1[12] : (!llvm.ptr) -> !llvm.ptr, f32
    %113 = llvm.load %112 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %114 = llvm.getelementptr inbounds %53[4] : (!llvm.ptr) -> !llvm.ptr, f32
    %115 = llvm.load %114 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %116 = llvm.call @__catalyst__rt__array_get_element_ptr_1d(%73, %3) : (!llvm.ptr, i64) -> !llvm.ptr
    %117 = llvm.load %116 : !llvm.ptr -> !llvm.ptr
    %118 = llvm.fpext %115 : f32 to f64
    llvm.call @__catalyst__qis__RY(%118, %117, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %119 = llvm.fpext %113 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%119, %117, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %120 = llvm.fpext %111 : f32 to f64
    llvm.call @__catalyst__qis__RY(%120, %117, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %121 = llvm.fpext %109 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%121, %117, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %122 = llvm.getelementptr inbounds %arg1[11] : (!llvm.ptr) -> !llvm.ptr, f32
    %123 = llvm.load %122 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %124 = llvm.getelementptr inbounds %arg1[10] : (!llvm.ptr) -> !llvm.ptr, f32
    %125 = llvm.load %124 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %126 = llvm.getelementptr inbounds %arg1[9] : (!llvm.ptr) -> !llvm.ptr, f32
    %127 = llvm.load %126 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %128 = llvm.getelementptr inbounds %53[3] : (!llvm.ptr) -> !llvm.ptr, f32
    %129 = llvm.load %128 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %130 = llvm.call @__catalyst__rt__array_get_element_ptr_1d(%73, %2) : (!llvm.ptr, i64) -> !llvm.ptr
    %131 = llvm.load %130 : !llvm.ptr -> !llvm.ptr
    %132 = llvm.fpext %129 : f32 to f64
    llvm.call @__catalyst__qis__RY(%132, %131, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %133 = llvm.fpext %127 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%133, %131, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %134 = llvm.fpext %125 : f32 to f64
    llvm.call @__catalyst__qis__RY(%134, %131, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %135 = llvm.fpext %123 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%135, %131, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %136 = llvm.getelementptr inbounds %arg1[8] : (!llvm.ptr) -> !llvm.ptr, f32
    %137 = llvm.load %136 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %138 = llvm.getelementptr inbounds %arg1[7] : (!llvm.ptr) -> !llvm.ptr, f32
    %139 = llvm.load %138 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %140 = llvm.getelementptr inbounds %arg1[6] : (!llvm.ptr) -> !llvm.ptr, f32
    %141 = llvm.load %140 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %142 = llvm.getelementptr inbounds %53[2] : (!llvm.ptr) -> !llvm.ptr, f32
    %143 = llvm.load %142 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %144 = llvm.call @__catalyst__rt__array_get_element_ptr_1d(%73, %1) : (!llvm.ptr, i64) -> !llvm.ptr
    %145 = llvm.load %144 : !llvm.ptr -> !llvm.ptr
    %146 = llvm.fpext %143 : f32 to f64
    llvm.call @__catalyst__qis__RY(%146, %145, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %147 = llvm.fpext %141 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%147, %145, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %148 = llvm.fpext %139 : f32 to f64
    llvm.call @__catalyst__qis__RY(%148, %145, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %149 = llvm.fpext %137 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%149, %145, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %150 = llvm.getelementptr inbounds %arg1[2] : (!llvm.ptr) -> !llvm.ptr, f32
    %151 = llvm.load %150 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %152 = llvm.getelementptr inbounds %arg1[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %153 = llvm.load %152 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %154 = llvm.load %arg1 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %155 = llvm.load %53 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %156 = llvm.call @__catalyst__rt__array_get_element_ptr_1d(%73, %16) : (!llvm.ptr, i64) -> !llvm.ptr
    %157 = llvm.load %156 : !llvm.ptr -> !llvm.ptr
    %158 = llvm.fpext %155 : f32 to f64
    llvm.call @__catalyst__qis__RY(%158, %157, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %159 = llvm.fpext %154 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%159, %157, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %160 = llvm.fpext %153 : f32 to f64
    llvm.call @__catalyst__qis__RY(%160, %157, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %161 = llvm.fpext %151 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%161, %157, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %162 = llvm.getelementptr inbounds %arg1[5] : (!llvm.ptr) -> !llvm.ptr, f32
    %163 = llvm.load %162 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %164 = llvm.getelementptr inbounds %arg1[4] : (!llvm.ptr) -> !llvm.ptr, f32
    %165 = llvm.load %164 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %166 = llvm.getelementptr inbounds %arg1[3] : (!llvm.ptr) -> !llvm.ptr, f32
    %167 = llvm.load %166 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %168 = llvm.getelementptr inbounds %53[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %169 = llvm.load %168 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    llvm.call @_mlir_memref_to_llvm_free(%47) : (!llvm.ptr) -> ()
    %170 = llvm.call @__catalyst__rt__array_get_element_ptr_1d(%73, %0) : (!llvm.ptr, i64) -> !llvm.ptr
    %171 = llvm.load %170 : !llvm.ptr -> !llvm.ptr
    %172 = llvm.fpext %169 : f32 to f64
    llvm.call @__catalyst__qis__RY(%172, %171, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %173 = llvm.fpext %167 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%173, %171, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %174 = llvm.fpext %165 : f32 to f64
    llvm.call @__catalyst__qis__RY(%174, %171, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %175 = llvm.fpext %163 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%175, %171, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%157, %171, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%171, %145, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%145, %131, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%131, %117, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%117, %103, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%103, %89, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%89, %75, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %176 = llvm.fpext %37 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%176, %89, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %177 = llvm.fpext %35 : f32 to f64
    llvm.call @__catalyst__qis__RY(%177, %89, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %178 = llvm.fpext %33 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%178, %89, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %179 = llvm.getelementptr inbounds %arg1[38] : (!llvm.ptr) -> !llvm.ptr, f32
    %180 = llvm.load %179 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %181 = llvm.getelementptr inbounds %arg1[37] : (!llvm.ptr) -> !llvm.ptr, f32
    %182 = llvm.load %181 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %183 = llvm.getelementptr inbounds %arg1[36] : (!llvm.ptr) -> !llvm.ptr, f32
    %184 = llvm.load %183 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %185 = llvm.fpext %184 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%185, %117, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %186 = llvm.fpext %182 : f32 to f64
    llvm.call @__catalyst__qis__RY(%186, %117, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %187 = llvm.fpext %180 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%187, %117, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %188 = llvm.getelementptr inbounds %arg1[26] : (!llvm.ptr) -> !llvm.ptr, f32
    %189 = llvm.load %188 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %190 = llvm.getelementptr inbounds %arg1[25] : (!llvm.ptr) -> !llvm.ptr, f32
    %191 = llvm.load %190 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %192 = llvm.getelementptr inbounds %arg1[24] : (!llvm.ptr) -> !llvm.ptr, f32
    %193 = llvm.load %192 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    llvm.call @__catalyst__qis__CNOT(%75, %157, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %194 = llvm.fpext %193 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%194, %157, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %195 = llvm.fpext %191 : f32 to f64
    llvm.call @__catalyst__qis__RY(%195, %157, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %196 = llvm.fpext %189 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%196, %157, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %197 = llvm.getelementptr inbounds %arg1[32] : (!llvm.ptr) -> !llvm.ptr, f32
    %198 = llvm.load %197 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %199 = llvm.getelementptr inbounds %arg1[31] : (!llvm.ptr) -> !llvm.ptr, f32
    %200 = llvm.load %199 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %201 = llvm.getelementptr inbounds %arg1[30] : (!llvm.ptr) -> !llvm.ptr, f32
    %202 = llvm.load %201 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %203 = llvm.fpext %202 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%203, %145, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %204 = llvm.fpext %200 : f32 to f64
    llvm.call @__catalyst__qis__RY(%204, %145, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %205 = llvm.fpext %198 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%205, %145, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%157, %145, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%145, %117, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%117, %89, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%89, %157, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %206 = llvm.fpext %31 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%206, %157, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %207 = llvm.fpext %29 : f32 to f64
    llvm.call @__catalyst__qis__RY(%207, %157, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %208 = llvm.fpext %27 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%208, %157, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %209 = llvm.getelementptr inbounds %arg1[59] : (!llvm.ptr) -> !llvm.ptr, f32
    %210 = llvm.load %209 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %211 = llvm.getelementptr inbounds %arg1[58] : (!llvm.ptr) -> !llvm.ptr, f32
    %212 = llvm.load %211 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %213 = llvm.getelementptr inbounds %arg1[57] : (!llvm.ptr) -> !llvm.ptr, f32
    %214 = llvm.load %213 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %215 = llvm.getelementptr inbounds %arg1[41] : (!llvm.ptr) -> !llvm.ptr, f32
    %216 = llvm.load %215 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %217 = llvm.getelementptr inbounds %arg1[40] : (!llvm.ptr) -> !llvm.ptr, f32
    %218 = llvm.load %217 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %219 = llvm.getelementptr inbounds %arg1[39] : (!llvm.ptr) -> !llvm.ptr, f32
    %220 = llvm.load %219 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %221 = llvm.fpext %220 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%221, %103, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %222 = llvm.fpext %218 : f32 to f64
    llvm.call @__catalyst__qis__RY(%222, %103, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %223 = llvm.fpext %216 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%223, %103, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %224 = llvm.getelementptr inbounds %arg1[29] : (!llvm.ptr) -> !llvm.ptr, f32
    %225 = llvm.load %224 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %226 = llvm.getelementptr inbounds %arg1[28] : (!llvm.ptr) -> !llvm.ptr, f32
    %227 = llvm.load %226 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %228 = llvm.getelementptr inbounds %arg1[27] : (!llvm.ptr) -> !llvm.ptr, f32
    %229 = llvm.load %228 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %230 = llvm.fpext %229 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%230, %171, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %231 = llvm.fpext %227 : f32 to f64
    llvm.call @__catalyst__qis__RY(%231, %171, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %232 = llvm.fpext %225 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%232, %171, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %233 = llvm.getelementptr inbounds %arg1[35] : (!llvm.ptr) -> !llvm.ptr, f32
    %234 = llvm.load %233 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %235 = llvm.getelementptr inbounds %arg1[34] : (!llvm.ptr) -> !llvm.ptr, f32
    %236 = llvm.load %235 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %237 = llvm.getelementptr inbounds %arg1[33] : (!llvm.ptr) -> !llvm.ptr, f32
    %238 = llvm.load %237 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %239 = llvm.fpext %238 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%239, %131, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %240 = llvm.fpext %236 : f32 to f64
    llvm.call @__catalyst__qis__RY(%240, %131, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %241 = llvm.fpext %234 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%241, %131, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%171, %131, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%131, %103, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %242 = llvm.fpext %214 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%242, %131, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %243 = llvm.fpext %212 : f32 to f64
    llvm.call @__catalyst__qis__RY(%243, %131, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %244 = llvm.fpext %210 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%244, %131, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%157, %131, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %245 = llvm.getelementptr inbounds %arg1[56] : (!llvm.ptr) -> !llvm.ptr, f32
    %246 = llvm.load %245 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %247 = llvm.getelementptr inbounds %arg1[55] : (!llvm.ptr) -> !llvm.ptr, f32
    %248 = llvm.load %247 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %249 = llvm.getelementptr inbounds %arg1[54] : (!llvm.ptr) -> !llvm.ptr, f32
    %250 = llvm.load %249 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %251 = llvm.fpext %250 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%251, %145, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %252 = llvm.fpext %248 : f32 to f64
    llvm.call @__catalyst__qis__RY(%252, %145, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %253 = llvm.fpext %246 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%253, %145, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %254 = llvm.getelementptr inbounds %arg1[65] : (!llvm.ptr) -> !llvm.ptr, f32
    %255 = llvm.load %254 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %256 = llvm.getelementptr inbounds %arg1[64] : (!llvm.ptr) -> !llvm.ptr, f32
    %257 = llvm.load %256 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %258 = llvm.getelementptr inbounds %arg1[63] : (!llvm.ptr) -> !llvm.ptr, f32
    %259 = llvm.load %258 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %260 = llvm.getelementptr inbounds %arg1[47] : (!llvm.ptr) -> !llvm.ptr, f32
    %261 = llvm.load %260 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %262 = llvm.getelementptr inbounds %arg1[46] : (!llvm.ptr) -> !llvm.ptr, f32
    %263 = llvm.load %262 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %264 = llvm.getelementptr inbounds %arg1[45] : (!llvm.ptr) -> !llvm.ptr, f32
    %265 = llvm.load %264 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %266 = llvm.fpext %265 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%266, %75, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %267 = llvm.fpext %263 : f32 to f64
    llvm.call @__catalyst__qis__RY(%267, %75, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %268 = llvm.fpext %261 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%268, %75, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%103, %75, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %269 = llvm.fpext %259 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%269, %103, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %270 = llvm.fpext %257 : f32 to f64
    llvm.call @__catalyst__qis__RY(%270, %103, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %271 = llvm.fpext %255 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%271, %103, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%145, %103, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%103, %157, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %272 = llvm.fpext %25 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%272, %157, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %273 = llvm.fpext %23 : f32 to f64
    llvm.call @__catalyst__qis__RY(%273, %157, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %274 = llvm.fpext %21 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%274, %157, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %275 = llvm.getelementptr inbounds %arg1[86] : (!llvm.ptr) -> !llvm.ptr, f32
    %276 = llvm.load %275 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %277 = llvm.getelementptr inbounds %arg1[85] : (!llvm.ptr) -> !llvm.ptr, f32
    %278 = llvm.load %277 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %279 = llvm.getelementptr inbounds %arg1[84] : (!llvm.ptr) -> !llvm.ptr, f32
    %280 = llvm.load %279 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %281 = llvm.getelementptr inbounds %arg1[71] : (!llvm.ptr) -> !llvm.ptr, f32
    %282 = llvm.load %281 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %283 = llvm.getelementptr inbounds %arg1[70] : (!llvm.ptr) -> !llvm.ptr, f32
    %284 = llvm.load %283 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %285 = llvm.getelementptr inbounds %arg1[69] : (!llvm.ptr) -> !llvm.ptr, f32
    %286 = llvm.load %285 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    llvm.call @__catalyst__qis__CNOT(%75, %171, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %287 = llvm.fpext %286 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%287, %75, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %288 = llvm.fpext %284 : f32 to f64
    llvm.call @__catalyst__qis__RY(%288, %75, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %289 = llvm.fpext %282 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%289, %75, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %290 = llvm.getelementptr inbounds %arg1[53] : (!llvm.ptr) -> !llvm.ptr, f32
    %291 = llvm.load %290 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %292 = llvm.getelementptr inbounds %arg1[52] : (!llvm.ptr) -> !llvm.ptr, f32
    %293 = llvm.load %292 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %294 = llvm.getelementptr inbounds %arg1[51] : (!llvm.ptr) -> !llvm.ptr, f32
    %295 = llvm.load %294 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %296 = llvm.fpext %295 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%296, %171, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %297 = llvm.fpext %293 : f32 to f64
    llvm.call @__catalyst__qis__RY(%297, %171, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %298 = llvm.fpext %291 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%298, %171, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %299 = llvm.getelementptr inbounds %arg1[62] : (!llvm.ptr) -> !llvm.ptr, f32
    %300 = llvm.load %299 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %301 = llvm.getelementptr inbounds %arg1[61] : (!llvm.ptr) -> !llvm.ptr, f32
    %302 = llvm.load %301 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %303 = llvm.getelementptr inbounds %arg1[60] : (!llvm.ptr) -> !llvm.ptr, f32
    %304 = llvm.load %303 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %305 = llvm.fpext %304 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%305, %117, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %306 = llvm.fpext %302 : f32 to f64
    llvm.call @__catalyst__qis__RY(%306, %117, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %307 = llvm.fpext %300 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%307, %117, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%171, %117, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%117, %75, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %308 = llvm.fpext %280 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%308, %117, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %309 = llvm.fpext %278 : f32 to f64
    llvm.call @__catalyst__qis__RY(%309, %117, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %310 = llvm.fpext %276 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%310, %117, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%157, %117, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%117, %157, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %311 = llvm.getelementptr inbounds %arg1[77] : (!llvm.ptr) -> !llvm.ptr, f32
    %312 = llvm.load %311 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %313 = llvm.getelementptr inbounds %arg1[76] : (!llvm.ptr) -> !llvm.ptr, f32
    %314 = llvm.load %313 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %315 = llvm.getelementptr inbounds %arg1[75] : (!llvm.ptr) -> !llvm.ptr, f32
    %316 = llvm.load %315 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %317 = llvm.getelementptr inbounds %arg1[68] : (!llvm.ptr) -> !llvm.ptr, f32
    %318 = llvm.load %317 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %319 = llvm.getelementptr inbounds %arg1[67] : (!llvm.ptr) -> !llvm.ptr, f32
    %320 = llvm.load %319 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %321 = llvm.getelementptr inbounds %arg1[66] : (!llvm.ptr) -> !llvm.ptr, f32
    %322 = llvm.load %321 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %323 = llvm.fpext %322 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%323, %89, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %324 = llvm.fpext %320 : f32 to f64
    llvm.call @__catalyst__qis__RY(%324, %89, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %325 = llvm.fpext %318 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%325, %89, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%131, %89, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%89, %171, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %326 = llvm.fpext %316 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%326, %171, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %327 = llvm.fpext %314 : f32 to f64
    llvm.call @__catalyst__qis__RY(%327, %171, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %328 = llvm.fpext %312 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%328, %171, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %329 = llvm.getelementptr inbounds %arg1[89] : (!llvm.ptr) -> !llvm.ptr, f32
    %330 = llvm.load %329 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %331 = llvm.getelementptr inbounds %arg1[88] : (!llvm.ptr) -> !llvm.ptr, f32
    %332 = llvm.load %331 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %333 = llvm.getelementptr inbounds %arg1[87] : (!llvm.ptr) -> !llvm.ptr, f32
    %334 = llvm.load %333 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %335 = llvm.fpext %334 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%335, %103, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %336 = llvm.fpext %332 : f32 to f64
    llvm.call @__catalyst__qis__RY(%336, %103, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %337 = llvm.fpext %330 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%337, %103, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%171, %103, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%103, %171, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %338 = llvm.getelementptr inbounds %arg1[80] : (!llvm.ptr) -> !llvm.ptr, f32
    %339 = llvm.load %338 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %340 = llvm.getelementptr inbounds %arg1[79] : (!llvm.ptr) -> !llvm.ptr, f32
    %341 = llvm.load %340 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %342 = llvm.getelementptr inbounds %arg1[78] : (!llvm.ptr) -> !llvm.ptr, f32
    %343 = llvm.load %342 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    llvm.call @__catalyst__qis__CNOT(%75, %145, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %344 = llvm.fpext %343 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%344, %145, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %345 = llvm.fpext %341 : f32 to f64
    llvm.call @__catalyst__qis__RY(%345, %145, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %346 = llvm.fpext %339 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%346, %145, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %347 = llvm.getelementptr inbounds %arg1[92] : (!llvm.ptr) -> !llvm.ptr, f32
    %348 = llvm.load %347 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %349 = llvm.getelementptr inbounds %arg1[91] : (!llvm.ptr) -> !llvm.ptr, f32
    %350 = llvm.load %349 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %351 = llvm.getelementptr inbounds %arg1[90] : (!llvm.ptr) -> !llvm.ptr, f32
    %352 = llvm.load %351 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %353 = llvm.fpext %352 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%353, %89, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %354 = llvm.fpext %350 : f32 to f64
    llvm.call @__catalyst__qis__RY(%354, %89, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %355 = llvm.fpext %348 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%355, %89, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%145, %89, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%89, %145, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %356 = llvm.getelementptr inbounds %arg1[83] : (!llvm.ptr) -> !llvm.ptr, f32
    %357 = llvm.load %356 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %358 = llvm.getelementptr inbounds %arg1[82] : (!llvm.ptr) -> !llvm.ptr, f32
    %359 = llvm.load %358 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %360 = llvm.getelementptr inbounds %arg1[81] : (!llvm.ptr) -> !llvm.ptr, f32
    %361 = llvm.load %360 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %362 = llvm.fpext %361 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%362, %131, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %363 = llvm.fpext %359 : f32 to f64
    llvm.call @__catalyst__qis__RY(%363, %131, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %364 = llvm.fpext %357 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%364, %131, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %365 = llvm.getelementptr inbounds %arg1[95] : (!llvm.ptr) -> !llvm.ptr, f32
    %366 = llvm.load %365 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %367 = llvm.getelementptr inbounds %arg1[94] : (!llvm.ptr) -> !llvm.ptr, f32
    %368 = llvm.load %367 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %369 = llvm.getelementptr inbounds %arg1[93] : (!llvm.ptr) -> !llvm.ptr, f32
    %370 = llvm.load %369 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %371 = llvm.fpext %370 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%371, %75, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %372 = llvm.fpext %368 : f32 to f64
    llvm.call @__catalyst__qis__RY(%372, %75, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %373 = llvm.fpext %366 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%373, %75, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%131, %75, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%75, %131, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %374 = llvm.call @__catalyst__qis__NamedObs(%2, %157) : (i64, !llvm.ptr) -> i64
    %375 = llvm.call @__catalyst__qis__Expval(%374) : (i64) -> f64
    %376 = llvm.getelementptr %15[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %377 = llvm.ptrtoint %376 : !llvm.ptr to i64
    %378 = llvm.add %377, %12 : i64
    %379 = llvm.call @_mlir_memref_to_llvm_alloc(%378) : (i64) -> !llvm.ptr
    %380 = llvm.ptrtoint %379 : !llvm.ptr to i64
    %381 = llvm.sub %12, %19 : i64
    %382 = llvm.add %380, %381 : i64
    %383 = llvm.urem %382, %12 : i64
    %384 = llvm.sub %382, %383 : i64
    %385 = llvm.inttoptr %384 : i64 to !llvm.ptr
    %386 = llvm.insertvalue %379, %13[0] : !llvm.struct<(ptr, ptr, i64)> 
    %387 = llvm.insertvalue %385, %386[1] : !llvm.struct<(ptr, ptr, i64)> 
    %388 = llvm.insertvalue %17, %387[2] : !llvm.struct<(ptr, ptr, i64)> 
    llvm.store %375, %385 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    llvm.call @__catalyst__rt__qubit_release_array(%73) : (!llvm.ptr) -> ()
    llvm.call @__catalyst__rt__device_release() : () -> ()
    llvm.return %388 : !llvm.struct<(ptr, ptr, i64)>
  }
  llvm.func @qnode_forward_0.adjoint(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: !llvm.ptr, %arg10: !llvm.ptr, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: i64) -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> attributes {sym_visibility = "private"} {
    %0 = llvm.mlir.constant(0 : index) : i64
    %1 = llvm.mlir.zero : !llvm.ptr
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.mlir.constant(true) : i1
    %4 = llvm.mlir.constant(false) : i1
    %5 = llvm.mlir.constant(1 : i64) : i64
    %6 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %7 = llvm.getelementptr %1[%arg14] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %8 = llvm.ptrtoint %7 : !llvm.ptr to i64
    %9 = llvm.call @_mlir_memref_to_llvm_alloc(%8) : (i64) -> !llvm.ptr
    %10 = llvm.insertvalue %9, %6[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %11 = llvm.insertvalue %9, %10[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %12 = llvm.insertvalue %0, %11[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %13 = llvm.insertvalue %arg14, %12[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %14 = llvm.insertvalue %2, %13[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.call @__catalyst__rt__toggle_recorder(%3) : (i1) -> ()
    %15 = llvm.call @qnode_forward_0.nodealloc(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> !llvm.struct<(ptr, f64)>
    %16 = llvm.extractvalue %15[0] : !llvm.struct<(ptr, f64)> 
    llvm.call @__catalyst__rt__toggle_recorder(%4) : (i1) -> ()
    %17 = llvm.alloca %5 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %14, %17 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    llvm.call @__catalyst__qis__Gradient(%5, %17) vararg(!llvm.func<void (i64, ...)>) : (i64, !llvm.ptr) -> ()
    llvm.call @__catalyst__rt__qubit_release_array(%16) : (!llvm.ptr) -> ()
    llvm.call @__catalyst__rt__device_release() : () -> ()
    llvm.return %14 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  }
  llvm.func @qnode_forward_0.nodealloc(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: !llvm.ptr, %arg10: !llvm.ptr, %arg11: i64, %arg12: i64, %arg13: i64) -> !llvm.struct<(ptr, f64)> attributes {sym_visibility = "private"} {
    %0 = llvm.mlir.poison : !llvm.struct<(ptr, f64)>
    %1 = llvm.mlir.constant(1 : i64) : i64
    %2 = llvm.mlir.constant(2 : i64) : i64
    %3 = llvm.mlir.constant(3 : i64) : i64
    %4 = llvm.mlir.constant(4 : i64) : i64
    %5 = llvm.mlir.constant(5 : i64) : i64
    %6 = llvm.mlir.constant(6 : i64) : i64
    %7 = llvm.mlir.constant(7 : i64) : i64
    %8 = llvm.mlir.constant(8 : i64) : i64
    %9 = llvm.mlir.constant(false) : i1
    %10 = llvm.mlir.addressof @"{}" : !llvm.ptr
    %11 = llvm.mlir.addressof @LightningGPUSimulator : !llvm.ptr
    %12 = llvm.mlir.addressof @"/home/pichau/QAgents/.venv/lib/python3.12/site-packages/pennylane_lightning/liblightning_gpu_catalyst.so" : !llvm.ptr
    %13 = llvm.mlir.constant(64 : index) : i64
    %14 = llvm.mlir.addressof @__constant_xf32 : !llvm.ptr
    %15 = llvm.mlir.zero : !llvm.ptr
    %16 = llvm.mlir.constant(0 : i64) : i64
    %17 = llvm.mlir.constant(0 : index) : i64
    %18 = llvm.mlir.constant(8 : index) : i64
    %19 = llvm.mlir.constant(1 : index) : i64
    %20 = llvm.getelementptr inbounds %arg1[74] : (!llvm.ptr) -> !llvm.ptr, f32
    %21 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %22 = llvm.getelementptr inbounds %arg1[73] : (!llvm.ptr) -> !llvm.ptr, f32
    %23 = llvm.load %22 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %24 = llvm.getelementptr inbounds %arg1[72] : (!llvm.ptr) -> !llvm.ptr, f32
    %25 = llvm.load %24 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %26 = llvm.getelementptr inbounds %arg1[50] : (!llvm.ptr) -> !llvm.ptr, f32
    %27 = llvm.load %26 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %28 = llvm.getelementptr inbounds %arg1[49] : (!llvm.ptr) -> !llvm.ptr, f32
    %29 = llvm.load %28 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %30 = llvm.getelementptr inbounds %arg1[48] : (!llvm.ptr) -> !llvm.ptr, f32
    %31 = llvm.load %30 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %32 = llvm.getelementptr inbounds %arg1[44] : (!llvm.ptr) -> !llvm.ptr, f32
    %33 = llvm.load %32 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %34 = llvm.getelementptr inbounds %arg1[43] : (!llvm.ptr) -> !llvm.ptr, f32
    %35 = llvm.load %34 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %36 = llvm.getelementptr inbounds %arg1[42] : (!llvm.ptr) -> !llvm.ptr, f32
    %37 = llvm.load %36 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %38 = llvm.getelementptr inbounds %arg1[23] : (!llvm.ptr) -> !llvm.ptr, f32
    %39 = llvm.load %38 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %40 = llvm.getelementptr inbounds %arg1[22] : (!llvm.ptr) -> !llvm.ptr, f32
    %41 = llvm.load %40 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %42 = llvm.getelementptr inbounds %arg1[21] : (!llvm.ptr) -> !llvm.ptr, f32
    %43 = llvm.load %42 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %44 = llvm.getelementptr %15[8] : (!llvm.ptr) -> !llvm.ptr, f32
    %45 = llvm.ptrtoint %44 : !llvm.ptr to i64
    %46 = llvm.add %45, %13 : i64
    %47 = llvm.call @_mlir_memref_to_llvm_alloc(%46) : (i64) -> !llvm.ptr
    %48 = llvm.ptrtoint %47 : !llvm.ptr to i64
    %49 = llvm.sub %13, %19 : i64
    %50 = llvm.add %48, %49 : i64
    %51 = llvm.urem %50, %13 : i64
    %52 = llvm.sub %50, %51 : i64
    %53 = llvm.inttoptr %52 : i64 to !llvm.ptr
    llvm.br ^bb1(%17 : i64)
  ^bb1(%54: i64):  // 2 preds: ^bb0, ^bb2
    %55 = llvm.icmp "slt" %54, %18 : i64
    llvm.cond_br %55, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %56 = llvm.load %14 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %57 = llvm.getelementptr inbounds %53[%54] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %56, %57 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f32, !llvm.ptr
    %58 = llvm.add %54, %19 : i64
    llvm.br ^bb1(%58 : i64)
  ^bb3:  // pred: ^bb1
    llvm.br ^bb4(%17 : i64)
  ^bb4(%59: i64):  // 2 preds: ^bb3, ^bb5
    %60 = llvm.icmp "slt" %59, %18 : i64
    llvm.cond_br %60, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %61 = llvm.getelementptr inbounds %53[%59] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %62 = llvm.load %61 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %63 = llvm.getelementptr inbounds %arg10[%59] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %64 = llvm.load %63 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %65 = llvm.fmul %62, %64 : f32
    %66 = llvm.getelementptr inbounds %53[%59] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %65, %66 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f32, !llvm.ptr
    %67 = llvm.add %59, %19 : i64
    llvm.br ^bb4(%67 : i64)
  ^bb6:  // pred: ^bb4
    %68 = llvm.getelementptr inbounds %53[7] : (!llvm.ptr) -> !llvm.ptr, f32
    %69 = llvm.load %68 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %70 = llvm.getelementptr inbounds %12[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<105 x i8>
    %71 = llvm.getelementptr inbounds %11[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<22 x i8>
    %72 = llvm.getelementptr inbounds %10[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3 x i8>
    llvm.call @__catalyst__rt__device_init(%70, %71, %72, %16, %9) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i1) -> ()
    %73 = llvm.call @__catalyst__rt__qubit_allocate_array(%8) : (i64) -> !llvm.ptr
    %74 = llvm.call @__catalyst__rt__array_get_element_ptr_1d(%73, %7) : (!llvm.ptr, i64) -> !llvm.ptr
    %75 = llvm.load %74 : !llvm.ptr -> !llvm.ptr
    %76 = llvm.fpext %69 : f32 to f64
    llvm.call @__catalyst__qis__RY(%76, %75, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %77 = llvm.fpext %43 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%77, %75, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %78 = llvm.fpext %41 : f32 to f64
    llvm.call @__catalyst__qis__RY(%78, %75, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %79 = llvm.fpext %39 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%79, %75, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %80 = llvm.getelementptr inbounds %arg1[20] : (!llvm.ptr) -> !llvm.ptr, f32
    %81 = llvm.load %80 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %82 = llvm.getelementptr inbounds %arg1[19] : (!llvm.ptr) -> !llvm.ptr, f32
    %83 = llvm.load %82 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %84 = llvm.getelementptr inbounds %arg1[18] : (!llvm.ptr) -> !llvm.ptr, f32
    %85 = llvm.load %84 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %86 = llvm.getelementptr inbounds %53[6] : (!llvm.ptr) -> !llvm.ptr, f32
    %87 = llvm.load %86 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %88 = llvm.call @__catalyst__rt__array_get_element_ptr_1d(%73, %6) : (!llvm.ptr, i64) -> !llvm.ptr
    %89 = llvm.load %88 : !llvm.ptr -> !llvm.ptr
    %90 = llvm.fpext %87 : f32 to f64
    llvm.call @__catalyst__qis__RY(%90, %89, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %91 = llvm.fpext %85 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%91, %89, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %92 = llvm.fpext %83 : f32 to f64
    llvm.call @__catalyst__qis__RY(%92, %89, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %93 = llvm.fpext %81 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%93, %89, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %94 = llvm.getelementptr inbounds %arg1[17] : (!llvm.ptr) -> !llvm.ptr, f32
    %95 = llvm.load %94 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %96 = llvm.getelementptr inbounds %arg1[16] : (!llvm.ptr) -> !llvm.ptr, f32
    %97 = llvm.load %96 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %98 = llvm.getelementptr inbounds %arg1[15] : (!llvm.ptr) -> !llvm.ptr, f32
    %99 = llvm.load %98 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %100 = llvm.getelementptr inbounds %53[5] : (!llvm.ptr) -> !llvm.ptr, f32
    %101 = llvm.load %100 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %102 = llvm.call @__catalyst__rt__array_get_element_ptr_1d(%73, %5) : (!llvm.ptr, i64) -> !llvm.ptr
    %103 = llvm.load %102 : !llvm.ptr -> !llvm.ptr
    %104 = llvm.fpext %101 : f32 to f64
    llvm.call @__catalyst__qis__RY(%104, %103, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %105 = llvm.fpext %99 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%105, %103, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %106 = llvm.fpext %97 : f32 to f64
    llvm.call @__catalyst__qis__RY(%106, %103, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %107 = llvm.fpext %95 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%107, %103, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %108 = llvm.getelementptr inbounds %arg1[14] : (!llvm.ptr) -> !llvm.ptr, f32
    %109 = llvm.load %108 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %110 = llvm.getelementptr inbounds %arg1[13] : (!llvm.ptr) -> !llvm.ptr, f32
    %111 = llvm.load %110 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %112 = llvm.getelementptr inbounds %arg1[12] : (!llvm.ptr) -> !llvm.ptr, f32
    %113 = llvm.load %112 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %114 = llvm.getelementptr inbounds %53[4] : (!llvm.ptr) -> !llvm.ptr, f32
    %115 = llvm.load %114 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %116 = llvm.call @__catalyst__rt__array_get_element_ptr_1d(%73, %4) : (!llvm.ptr, i64) -> !llvm.ptr
    %117 = llvm.load %116 : !llvm.ptr -> !llvm.ptr
    %118 = llvm.fpext %115 : f32 to f64
    llvm.call @__catalyst__qis__RY(%118, %117, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %119 = llvm.fpext %113 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%119, %117, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %120 = llvm.fpext %111 : f32 to f64
    llvm.call @__catalyst__qis__RY(%120, %117, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %121 = llvm.fpext %109 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%121, %117, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %122 = llvm.getelementptr inbounds %arg1[11] : (!llvm.ptr) -> !llvm.ptr, f32
    %123 = llvm.load %122 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %124 = llvm.getelementptr inbounds %arg1[10] : (!llvm.ptr) -> !llvm.ptr, f32
    %125 = llvm.load %124 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %126 = llvm.getelementptr inbounds %arg1[9] : (!llvm.ptr) -> !llvm.ptr, f32
    %127 = llvm.load %126 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %128 = llvm.getelementptr inbounds %53[3] : (!llvm.ptr) -> !llvm.ptr, f32
    %129 = llvm.load %128 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %130 = llvm.call @__catalyst__rt__array_get_element_ptr_1d(%73, %3) : (!llvm.ptr, i64) -> !llvm.ptr
    %131 = llvm.load %130 : !llvm.ptr -> !llvm.ptr
    %132 = llvm.fpext %129 : f32 to f64
    llvm.call @__catalyst__qis__RY(%132, %131, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %133 = llvm.fpext %127 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%133, %131, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %134 = llvm.fpext %125 : f32 to f64
    llvm.call @__catalyst__qis__RY(%134, %131, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %135 = llvm.fpext %123 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%135, %131, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %136 = llvm.getelementptr inbounds %arg1[8] : (!llvm.ptr) -> !llvm.ptr, f32
    %137 = llvm.load %136 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %138 = llvm.getelementptr inbounds %arg1[7] : (!llvm.ptr) -> !llvm.ptr, f32
    %139 = llvm.load %138 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %140 = llvm.getelementptr inbounds %arg1[6] : (!llvm.ptr) -> !llvm.ptr, f32
    %141 = llvm.load %140 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %142 = llvm.getelementptr inbounds %53[2] : (!llvm.ptr) -> !llvm.ptr, f32
    %143 = llvm.load %142 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %144 = llvm.call @__catalyst__rt__array_get_element_ptr_1d(%73, %2) : (!llvm.ptr, i64) -> !llvm.ptr
    %145 = llvm.load %144 : !llvm.ptr -> !llvm.ptr
    %146 = llvm.fpext %143 : f32 to f64
    llvm.call @__catalyst__qis__RY(%146, %145, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %147 = llvm.fpext %141 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%147, %145, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %148 = llvm.fpext %139 : f32 to f64
    llvm.call @__catalyst__qis__RY(%148, %145, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %149 = llvm.fpext %137 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%149, %145, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %150 = llvm.getelementptr inbounds %arg1[2] : (!llvm.ptr) -> !llvm.ptr, f32
    %151 = llvm.load %150 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %152 = llvm.getelementptr inbounds %arg1[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %153 = llvm.load %152 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %154 = llvm.load %arg1 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %155 = llvm.load %53 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %156 = llvm.call @__catalyst__rt__array_get_element_ptr_1d(%73, %16) : (!llvm.ptr, i64) -> !llvm.ptr
    %157 = llvm.load %156 : !llvm.ptr -> !llvm.ptr
    %158 = llvm.fpext %155 : f32 to f64
    llvm.call @__catalyst__qis__RY(%158, %157, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %159 = llvm.fpext %154 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%159, %157, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %160 = llvm.fpext %153 : f32 to f64
    llvm.call @__catalyst__qis__RY(%160, %157, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %161 = llvm.fpext %151 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%161, %157, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %162 = llvm.getelementptr inbounds %arg1[5] : (!llvm.ptr) -> !llvm.ptr, f32
    %163 = llvm.load %162 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %164 = llvm.getelementptr inbounds %arg1[4] : (!llvm.ptr) -> !llvm.ptr, f32
    %165 = llvm.load %164 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %166 = llvm.getelementptr inbounds %arg1[3] : (!llvm.ptr) -> !llvm.ptr, f32
    %167 = llvm.load %166 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %168 = llvm.getelementptr inbounds %53[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %169 = llvm.load %168 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    llvm.call @_mlir_memref_to_llvm_free(%47) : (!llvm.ptr) -> ()
    %170 = llvm.call @__catalyst__rt__array_get_element_ptr_1d(%73, %1) : (!llvm.ptr, i64) -> !llvm.ptr
    %171 = llvm.load %170 : !llvm.ptr -> !llvm.ptr
    %172 = llvm.fpext %169 : f32 to f64
    llvm.call @__catalyst__qis__RY(%172, %171, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %173 = llvm.fpext %167 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%173, %171, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %174 = llvm.fpext %165 : f32 to f64
    llvm.call @__catalyst__qis__RY(%174, %171, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %175 = llvm.fpext %163 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%175, %171, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%157, %171, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%171, %145, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%145, %131, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%131, %117, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%117, %103, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%103, %89, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%89, %75, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %176 = llvm.fpext %37 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%176, %89, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %177 = llvm.fpext %35 : f32 to f64
    llvm.call @__catalyst__qis__RY(%177, %89, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %178 = llvm.fpext %33 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%178, %89, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %179 = llvm.getelementptr inbounds %arg1[38] : (!llvm.ptr) -> !llvm.ptr, f32
    %180 = llvm.load %179 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %181 = llvm.getelementptr inbounds %arg1[37] : (!llvm.ptr) -> !llvm.ptr, f32
    %182 = llvm.load %181 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %183 = llvm.getelementptr inbounds %arg1[36] : (!llvm.ptr) -> !llvm.ptr, f32
    %184 = llvm.load %183 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %185 = llvm.fpext %184 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%185, %117, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %186 = llvm.fpext %182 : f32 to f64
    llvm.call @__catalyst__qis__RY(%186, %117, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %187 = llvm.fpext %180 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%187, %117, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %188 = llvm.getelementptr inbounds %arg1[26] : (!llvm.ptr) -> !llvm.ptr, f32
    %189 = llvm.load %188 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %190 = llvm.getelementptr inbounds %arg1[25] : (!llvm.ptr) -> !llvm.ptr, f32
    %191 = llvm.load %190 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %192 = llvm.getelementptr inbounds %arg1[24] : (!llvm.ptr) -> !llvm.ptr, f32
    %193 = llvm.load %192 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    llvm.call @__catalyst__qis__CNOT(%75, %157, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %194 = llvm.fpext %193 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%194, %157, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %195 = llvm.fpext %191 : f32 to f64
    llvm.call @__catalyst__qis__RY(%195, %157, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %196 = llvm.fpext %189 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%196, %157, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %197 = llvm.getelementptr inbounds %arg1[32] : (!llvm.ptr) -> !llvm.ptr, f32
    %198 = llvm.load %197 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %199 = llvm.getelementptr inbounds %arg1[31] : (!llvm.ptr) -> !llvm.ptr, f32
    %200 = llvm.load %199 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %201 = llvm.getelementptr inbounds %arg1[30] : (!llvm.ptr) -> !llvm.ptr, f32
    %202 = llvm.load %201 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %203 = llvm.fpext %202 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%203, %145, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %204 = llvm.fpext %200 : f32 to f64
    llvm.call @__catalyst__qis__RY(%204, %145, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %205 = llvm.fpext %198 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%205, %145, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%157, %145, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%145, %117, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%117, %89, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%89, %157, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %206 = llvm.fpext %31 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%206, %157, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %207 = llvm.fpext %29 : f32 to f64
    llvm.call @__catalyst__qis__RY(%207, %157, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %208 = llvm.fpext %27 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%208, %157, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %209 = llvm.getelementptr inbounds %arg1[59] : (!llvm.ptr) -> !llvm.ptr, f32
    %210 = llvm.load %209 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %211 = llvm.getelementptr inbounds %arg1[58] : (!llvm.ptr) -> !llvm.ptr, f32
    %212 = llvm.load %211 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %213 = llvm.getelementptr inbounds %arg1[57] : (!llvm.ptr) -> !llvm.ptr, f32
    %214 = llvm.load %213 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %215 = llvm.getelementptr inbounds %arg1[41] : (!llvm.ptr) -> !llvm.ptr, f32
    %216 = llvm.load %215 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %217 = llvm.getelementptr inbounds %arg1[40] : (!llvm.ptr) -> !llvm.ptr, f32
    %218 = llvm.load %217 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %219 = llvm.getelementptr inbounds %arg1[39] : (!llvm.ptr) -> !llvm.ptr, f32
    %220 = llvm.load %219 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %221 = llvm.fpext %220 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%221, %103, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %222 = llvm.fpext %218 : f32 to f64
    llvm.call @__catalyst__qis__RY(%222, %103, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %223 = llvm.fpext %216 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%223, %103, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %224 = llvm.getelementptr inbounds %arg1[29] : (!llvm.ptr) -> !llvm.ptr, f32
    %225 = llvm.load %224 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %226 = llvm.getelementptr inbounds %arg1[28] : (!llvm.ptr) -> !llvm.ptr, f32
    %227 = llvm.load %226 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %228 = llvm.getelementptr inbounds %arg1[27] : (!llvm.ptr) -> !llvm.ptr, f32
    %229 = llvm.load %228 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %230 = llvm.fpext %229 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%230, %171, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %231 = llvm.fpext %227 : f32 to f64
    llvm.call @__catalyst__qis__RY(%231, %171, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %232 = llvm.fpext %225 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%232, %171, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %233 = llvm.getelementptr inbounds %arg1[35] : (!llvm.ptr) -> !llvm.ptr, f32
    %234 = llvm.load %233 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %235 = llvm.getelementptr inbounds %arg1[34] : (!llvm.ptr) -> !llvm.ptr, f32
    %236 = llvm.load %235 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %237 = llvm.getelementptr inbounds %arg1[33] : (!llvm.ptr) -> !llvm.ptr, f32
    %238 = llvm.load %237 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %239 = llvm.fpext %238 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%239, %131, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %240 = llvm.fpext %236 : f32 to f64
    llvm.call @__catalyst__qis__RY(%240, %131, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %241 = llvm.fpext %234 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%241, %131, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%171, %131, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%131, %103, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %242 = llvm.fpext %214 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%242, %131, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %243 = llvm.fpext %212 : f32 to f64
    llvm.call @__catalyst__qis__RY(%243, %131, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %244 = llvm.fpext %210 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%244, %131, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%157, %131, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %245 = llvm.getelementptr inbounds %arg1[56] : (!llvm.ptr) -> !llvm.ptr, f32
    %246 = llvm.load %245 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %247 = llvm.getelementptr inbounds %arg1[55] : (!llvm.ptr) -> !llvm.ptr, f32
    %248 = llvm.load %247 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %249 = llvm.getelementptr inbounds %arg1[54] : (!llvm.ptr) -> !llvm.ptr, f32
    %250 = llvm.load %249 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %251 = llvm.fpext %250 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%251, %145, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %252 = llvm.fpext %248 : f32 to f64
    llvm.call @__catalyst__qis__RY(%252, %145, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %253 = llvm.fpext %246 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%253, %145, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %254 = llvm.getelementptr inbounds %arg1[65] : (!llvm.ptr) -> !llvm.ptr, f32
    %255 = llvm.load %254 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %256 = llvm.getelementptr inbounds %arg1[64] : (!llvm.ptr) -> !llvm.ptr, f32
    %257 = llvm.load %256 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %258 = llvm.getelementptr inbounds %arg1[63] : (!llvm.ptr) -> !llvm.ptr, f32
    %259 = llvm.load %258 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %260 = llvm.getelementptr inbounds %arg1[47] : (!llvm.ptr) -> !llvm.ptr, f32
    %261 = llvm.load %260 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %262 = llvm.getelementptr inbounds %arg1[46] : (!llvm.ptr) -> !llvm.ptr, f32
    %263 = llvm.load %262 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %264 = llvm.getelementptr inbounds %arg1[45] : (!llvm.ptr) -> !llvm.ptr, f32
    %265 = llvm.load %264 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %266 = llvm.fpext %265 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%266, %75, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %267 = llvm.fpext %263 : f32 to f64
    llvm.call @__catalyst__qis__RY(%267, %75, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %268 = llvm.fpext %261 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%268, %75, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%103, %75, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %269 = llvm.fpext %259 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%269, %103, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %270 = llvm.fpext %257 : f32 to f64
    llvm.call @__catalyst__qis__RY(%270, %103, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %271 = llvm.fpext %255 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%271, %103, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%145, %103, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%103, %157, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %272 = llvm.fpext %25 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%272, %157, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %273 = llvm.fpext %23 : f32 to f64
    llvm.call @__catalyst__qis__RY(%273, %157, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %274 = llvm.fpext %21 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%274, %157, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %275 = llvm.getelementptr inbounds %arg1[86] : (!llvm.ptr) -> !llvm.ptr, f32
    %276 = llvm.load %275 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %277 = llvm.getelementptr inbounds %arg1[85] : (!llvm.ptr) -> !llvm.ptr, f32
    %278 = llvm.load %277 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %279 = llvm.getelementptr inbounds %arg1[84] : (!llvm.ptr) -> !llvm.ptr, f32
    %280 = llvm.load %279 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %281 = llvm.getelementptr inbounds %arg1[71] : (!llvm.ptr) -> !llvm.ptr, f32
    %282 = llvm.load %281 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %283 = llvm.getelementptr inbounds %arg1[70] : (!llvm.ptr) -> !llvm.ptr, f32
    %284 = llvm.load %283 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %285 = llvm.getelementptr inbounds %arg1[69] : (!llvm.ptr) -> !llvm.ptr, f32
    %286 = llvm.load %285 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    llvm.call @__catalyst__qis__CNOT(%75, %171, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %287 = llvm.fpext %286 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%287, %75, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %288 = llvm.fpext %284 : f32 to f64
    llvm.call @__catalyst__qis__RY(%288, %75, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %289 = llvm.fpext %282 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%289, %75, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %290 = llvm.getelementptr inbounds %arg1[53] : (!llvm.ptr) -> !llvm.ptr, f32
    %291 = llvm.load %290 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %292 = llvm.getelementptr inbounds %arg1[52] : (!llvm.ptr) -> !llvm.ptr, f32
    %293 = llvm.load %292 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %294 = llvm.getelementptr inbounds %arg1[51] : (!llvm.ptr) -> !llvm.ptr, f32
    %295 = llvm.load %294 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %296 = llvm.fpext %295 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%296, %171, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %297 = llvm.fpext %293 : f32 to f64
    llvm.call @__catalyst__qis__RY(%297, %171, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %298 = llvm.fpext %291 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%298, %171, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %299 = llvm.getelementptr inbounds %arg1[62] : (!llvm.ptr) -> !llvm.ptr, f32
    %300 = llvm.load %299 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %301 = llvm.getelementptr inbounds %arg1[61] : (!llvm.ptr) -> !llvm.ptr, f32
    %302 = llvm.load %301 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %303 = llvm.getelementptr inbounds %arg1[60] : (!llvm.ptr) -> !llvm.ptr, f32
    %304 = llvm.load %303 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %305 = llvm.fpext %304 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%305, %117, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %306 = llvm.fpext %302 : f32 to f64
    llvm.call @__catalyst__qis__RY(%306, %117, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %307 = llvm.fpext %300 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%307, %117, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%171, %117, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%117, %75, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %308 = llvm.fpext %280 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%308, %117, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %309 = llvm.fpext %278 : f32 to f64
    llvm.call @__catalyst__qis__RY(%309, %117, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %310 = llvm.fpext %276 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%310, %117, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%157, %117, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%117, %157, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %311 = llvm.getelementptr inbounds %arg1[77] : (!llvm.ptr) -> !llvm.ptr, f32
    %312 = llvm.load %311 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %313 = llvm.getelementptr inbounds %arg1[76] : (!llvm.ptr) -> !llvm.ptr, f32
    %314 = llvm.load %313 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %315 = llvm.getelementptr inbounds %arg1[75] : (!llvm.ptr) -> !llvm.ptr, f32
    %316 = llvm.load %315 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %317 = llvm.getelementptr inbounds %arg1[68] : (!llvm.ptr) -> !llvm.ptr, f32
    %318 = llvm.load %317 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %319 = llvm.getelementptr inbounds %arg1[67] : (!llvm.ptr) -> !llvm.ptr, f32
    %320 = llvm.load %319 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %321 = llvm.getelementptr inbounds %arg1[66] : (!llvm.ptr) -> !llvm.ptr, f32
    %322 = llvm.load %321 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %323 = llvm.fpext %322 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%323, %89, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %324 = llvm.fpext %320 : f32 to f64
    llvm.call @__catalyst__qis__RY(%324, %89, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %325 = llvm.fpext %318 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%325, %89, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%131, %89, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%89, %171, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %326 = llvm.fpext %316 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%326, %171, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %327 = llvm.fpext %314 : f32 to f64
    llvm.call @__catalyst__qis__RY(%327, %171, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %328 = llvm.fpext %312 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%328, %171, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %329 = llvm.getelementptr inbounds %arg1[89] : (!llvm.ptr) -> !llvm.ptr, f32
    %330 = llvm.load %329 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %331 = llvm.getelementptr inbounds %arg1[88] : (!llvm.ptr) -> !llvm.ptr, f32
    %332 = llvm.load %331 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %333 = llvm.getelementptr inbounds %arg1[87] : (!llvm.ptr) -> !llvm.ptr, f32
    %334 = llvm.load %333 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %335 = llvm.fpext %334 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%335, %103, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %336 = llvm.fpext %332 : f32 to f64
    llvm.call @__catalyst__qis__RY(%336, %103, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %337 = llvm.fpext %330 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%337, %103, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%171, %103, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%103, %171, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %338 = llvm.getelementptr inbounds %arg1[80] : (!llvm.ptr) -> !llvm.ptr, f32
    %339 = llvm.load %338 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %340 = llvm.getelementptr inbounds %arg1[79] : (!llvm.ptr) -> !llvm.ptr, f32
    %341 = llvm.load %340 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %342 = llvm.getelementptr inbounds %arg1[78] : (!llvm.ptr) -> !llvm.ptr, f32
    %343 = llvm.load %342 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    llvm.call @__catalyst__qis__CNOT(%75, %145, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %344 = llvm.fpext %343 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%344, %145, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %345 = llvm.fpext %341 : f32 to f64
    llvm.call @__catalyst__qis__RY(%345, %145, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %346 = llvm.fpext %339 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%346, %145, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %347 = llvm.getelementptr inbounds %arg1[92] : (!llvm.ptr) -> !llvm.ptr, f32
    %348 = llvm.load %347 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %349 = llvm.getelementptr inbounds %arg1[91] : (!llvm.ptr) -> !llvm.ptr, f32
    %350 = llvm.load %349 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %351 = llvm.getelementptr inbounds %arg1[90] : (!llvm.ptr) -> !llvm.ptr, f32
    %352 = llvm.load %351 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %353 = llvm.fpext %352 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%353, %89, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %354 = llvm.fpext %350 : f32 to f64
    llvm.call @__catalyst__qis__RY(%354, %89, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %355 = llvm.fpext %348 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%355, %89, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%145, %89, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%89, %145, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %356 = llvm.getelementptr inbounds %arg1[83] : (!llvm.ptr) -> !llvm.ptr, f32
    %357 = llvm.load %356 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %358 = llvm.getelementptr inbounds %arg1[82] : (!llvm.ptr) -> !llvm.ptr, f32
    %359 = llvm.load %358 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %360 = llvm.getelementptr inbounds %arg1[81] : (!llvm.ptr) -> !llvm.ptr, f32
    %361 = llvm.load %360 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %362 = llvm.fpext %361 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%362, %131, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %363 = llvm.fpext %359 : f32 to f64
    llvm.call @__catalyst__qis__RY(%363, %131, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %364 = llvm.fpext %357 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%364, %131, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %365 = llvm.getelementptr inbounds %arg1[95] : (!llvm.ptr) -> !llvm.ptr, f32
    %366 = llvm.load %365 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %367 = llvm.getelementptr inbounds %arg1[94] : (!llvm.ptr) -> !llvm.ptr, f32
    %368 = llvm.load %367 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %369 = llvm.getelementptr inbounds %arg1[93] : (!llvm.ptr) -> !llvm.ptr, f32
    %370 = llvm.load %369 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %371 = llvm.fpext %370 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%371, %75, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %372 = llvm.fpext %368 : f32 to f64
    llvm.call @__catalyst__qis__RY(%372, %75, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %373 = llvm.fpext %366 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%373, %75, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%131, %75, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%75, %131, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %374 = llvm.call @__catalyst__qis__NamedObs(%3, %157) : (i64, !llvm.ptr) -> i64
    %375 = llvm.call @__catalyst__qis__Expval(%374) : (i64) -> f64
    %376 = llvm.insertvalue %73, %0[0] : !llvm.struct<(ptr, f64)> 
    %377 = llvm.insertvalue %375, %376[1] : !llvm.struct<(ptr, f64)> 
    llvm.return %377 : !llvm.struct<(ptr, f64)>
  }
  llvm.func @qnode_forward_0.pcount(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: !llvm.ptr, %arg10: !llvm.ptr, %arg11: i64, %arg12: i64, %arg13: i64) -> i64 attributes {sym_visibility = "private"} {
    %0 = llvm.mlir.constant(1 : index) : i64
    %1 = llvm.mlir.constant(1 : i64) : i64
    %2 = llvm.mlir.constant(0 : i64) : i64
    %3 = llvm.alloca %0 x i64 : (i64) -> !llvm.ptr
    llvm.store %2, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %4 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %5 = llvm.add %4, %1 : i64
    llvm.store %5, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %6 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %7 = llvm.add %6, %1 : i64
    llvm.store %7, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %8 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %9 = llvm.add %8, %1 : i64
    llvm.store %9, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %10 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %11 = llvm.add %10, %1 : i64
    llvm.store %11, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %12 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %13 = llvm.add %12, %1 : i64
    llvm.store %13, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %14 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %15 = llvm.add %14, %1 : i64
    llvm.store %15, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %16 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %17 = llvm.add %16, %1 : i64
    llvm.store %17, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %18 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %19 = llvm.add %18, %1 : i64
    llvm.store %19, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %20 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %21 = llvm.add %20, %1 : i64
    llvm.store %21, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %22 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %23 = llvm.add %22, %1 : i64
    llvm.store %23, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %24 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %25 = llvm.add %24, %1 : i64
    llvm.store %25, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %26 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %27 = llvm.add %26, %1 : i64
    llvm.store %27, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %28 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %29 = llvm.add %28, %1 : i64
    llvm.store %29, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %30 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %31 = llvm.add %30, %1 : i64
    llvm.store %31, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %32 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %33 = llvm.add %32, %1 : i64
    llvm.store %33, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %34 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %35 = llvm.add %34, %1 : i64
    llvm.store %35, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %36 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %37 = llvm.add %36, %1 : i64
    llvm.store %37, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %38 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %39 = llvm.add %38, %1 : i64
    llvm.store %39, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %40 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %41 = llvm.add %40, %1 : i64
    llvm.store %41, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %42 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %43 = llvm.add %42, %1 : i64
    llvm.store %43, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %44 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %45 = llvm.add %44, %1 : i64
    llvm.store %45, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %46 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %47 = llvm.add %46, %1 : i64
    llvm.store %47, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %48 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %49 = llvm.add %48, %1 : i64
    llvm.store %49, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %50 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %51 = llvm.add %50, %1 : i64
    llvm.store %51, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %52 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %53 = llvm.add %52, %1 : i64
    llvm.store %53, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %54 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %55 = llvm.add %54, %1 : i64
    llvm.store %55, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %56 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %57 = llvm.add %56, %1 : i64
    llvm.store %57, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %58 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %59 = llvm.add %58, %1 : i64
    llvm.store %59, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %60 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %61 = llvm.add %60, %1 : i64
    llvm.store %61, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %62 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %63 = llvm.add %62, %1 : i64
    llvm.store %63, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %64 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %65 = llvm.add %64, %1 : i64
    llvm.store %65, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %66 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %67 = llvm.add %66, %1 : i64
    llvm.store %67, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %68 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %69 = llvm.add %68, %1 : i64
    llvm.store %69, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %70 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %71 = llvm.add %70, %1 : i64
    llvm.store %71, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %72 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %73 = llvm.add %72, %1 : i64
    llvm.store %73, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %74 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %75 = llvm.add %74, %1 : i64
    llvm.store %75, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %76 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %77 = llvm.add %76, %1 : i64
    llvm.store %77, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %78 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %79 = llvm.add %78, %1 : i64
    llvm.store %79, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %80 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %81 = llvm.add %80, %1 : i64
    llvm.store %81, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %82 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %83 = llvm.add %82, %1 : i64
    llvm.store %83, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %84 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %85 = llvm.add %84, %1 : i64
    llvm.store %85, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %86 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %87 = llvm.add %86, %1 : i64
    llvm.store %87, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %88 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %89 = llvm.add %88, %1 : i64
    llvm.store %89, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %90 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %91 = llvm.add %90, %1 : i64
    llvm.store %91, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %92 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %93 = llvm.add %92, %1 : i64
    llvm.store %93, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %94 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %95 = llvm.add %94, %1 : i64
    llvm.store %95, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %96 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %97 = llvm.add %96, %1 : i64
    llvm.store %97, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %98 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %99 = llvm.add %98, %1 : i64
    llvm.store %99, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %100 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %101 = llvm.add %100, %1 : i64
    llvm.store %101, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %102 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %103 = llvm.add %102, %1 : i64
    llvm.store %103, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %104 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %105 = llvm.add %104, %1 : i64
    llvm.store %105, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %106 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %107 = llvm.add %106, %1 : i64
    llvm.store %107, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %108 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %109 = llvm.add %108, %1 : i64
    llvm.store %109, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %110 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %111 = llvm.add %110, %1 : i64
    llvm.store %111, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %112 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %113 = llvm.add %112, %1 : i64
    llvm.store %113, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %114 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %115 = llvm.add %114, %1 : i64
    llvm.store %115, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %116 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %117 = llvm.add %116, %1 : i64
    llvm.store %117, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %118 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %119 = llvm.add %118, %1 : i64
    llvm.store %119, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %120 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %121 = llvm.add %120, %1 : i64
    llvm.store %121, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %122 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %123 = llvm.add %122, %1 : i64
    llvm.store %123, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %124 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %125 = llvm.add %124, %1 : i64
    llvm.store %125, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %126 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %127 = llvm.add %126, %1 : i64
    llvm.store %127, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %128 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %129 = llvm.add %128, %1 : i64
    llvm.store %129, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %130 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %131 = llvm.add %130, %1 : i64
    llvm.store %131, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %132 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %133 = llvm.add %132, %1 : i64
    llvm.store %133, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %134 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %135 = llvm.add %134, %1 : i64
    llvm.store %135, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %136 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %137 = llvm.add %136, %1 : i64
    llvm.store %137, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %138 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %139 = llvm.add %138, %1 : i64
    llvm.store %139, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %140 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %141 = llvm.add %140, %1 : i64
    llvm.store %141, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %142 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %143 = llvm.add %142, %1 : i64
    llvm.store %143, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %144 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %145 = llvm.add %144, %1 : i64
    llvm.store %145, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %146 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %147 = llvm.add %146, %1 : i64
    llvm.store %147, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %148 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %149 = llvm.add %148, %1 : i64
    llvm.store %149, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %150 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %151 = llvm.add %150, %1 : i64
    llvm.store %151, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %152 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %153 = llvm.add %152, %1 : i64
    llvm.store %153, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %154 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %155 = llvm.add %154, %1 : i64
    llvm.store %155, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %156 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %157 = llvm.add %156, %1 : i64
    llvm.store %157, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %158 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %159 = llvm.add %158, %1 : i64
    llvm.store %159, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %160 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %161 = llvm.add %160, %1 : i64
    llvm.store %161, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %162 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %163 = llvm.add %162, %1 : i64
    llvm.store %163, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %164 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %165 = llvm.add %164, %1 : i64
    llvm.store %165, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %166 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %167 = llvm.add %166, %1 : i64
    llvm.store %167, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %168 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %169 = llvm.add %168, %1 : i64
    llvm.store %169, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %170 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %171 = llvm.add %170, %1 : i64
    llvm.store %171, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %172 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %173 = llvm.add %172, %1 : i64
    llvm.store %173, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %174 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %175 = llvm.add %174, %1 : i64
    llvm.store %175, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %176 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %177 = llvm.add %176, %1 : i64
    llvm.store %177, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %178 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %179 = llvm.add %178, %1 : i64
    llvm.store %179, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %180 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %181 = llvm.add %180, %1 : i64
    llvm.store %181, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %182 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %183 = llvm.add %182, %1 : i64
    llvm.store %183, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %184 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %185 = llvm.add %184, %1 : i64
    llvm.store %185, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %186 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %187 = llvm.add %186, %1 : i64
    llvm.store %187, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %188 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %189 = llvm.add %188, %1 : i64
    llvm.store %189, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %190 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %191 = llvm.add %190, %1 : i64
    llvm.store %191, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %192 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %193 = llvm.add %192, %1 : i64
    llvm.store %193, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %194 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %195 = llvm.add %194, %1 : i64
    llvm.store %195, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %196 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %197 = llvm.add %196, %1 : i64
    llvm.store %197, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %198 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %199 = llvm.add %198, %1 : i64
    llvm.store %199, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %200 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %201 = llvm.add %200, %1 : i64
    llvm.store %201, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %202 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %203 = llvm.add %202, %1 : i64
    llvm.store %203, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %204 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %205 = llvm.add %204, %1 : i64
    llvm.store %205, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %206 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %207 = llvm.add %206, %1 : i64
    llvm.store %207, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %208 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %209 = llvm.add %208, %1 : i64
    llvm.store %209, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %210 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %211 = llvm.add %210, %1 : i64
    llvm.store %211, %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %212 = llvm.load %3 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    llvm.return %212 : i64
  }
  llvm.func @qnode_forward_0.quantum.customqgrad(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: !llvm.ptr, %arg6: !llvm.ptr, %arg7: !llvm.ptr, %arg8: !llvm.ptr) attributes {sym_visibility = "private"} {
    %0 = llvm.mlir.constant(1 : index) : i64
    %1 = llvm.mlir.constant(0 : index) : i64
    %2 = llvm.load %arg0 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %3 = llvm.load %arg2 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %4 = llvm.load %arg5 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %5 = llvm.extractvalue %4[3] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %6 = llvm.alloca %0 x !llvm.array<1 x i64> : (i64) -> !llvm.ptr
    llvm.store %5, %6 : !llvm.array<1 x i64>, !llvm.ptr
    %7 = llvm.getelementptr inbounds %6[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x i64>
    %8 = llvm.load %7 : !llvm.ptr -> i64
    %9 = llvm.extractvalue %2[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %10 = llvm.extractvalue %2[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %11 = llvm.extractvalue %2[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %12 = llvm.extractvalue %2[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %13 = llvm.extractvalue %2[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %14 = llvm.extractvalue %2[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %15 = llvm.extractvalue %2[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %16 = llvm.extractvalue %2[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %17 = llvm.extractvalue %2[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %18 = llvm.extractvalue %3[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %19 = llvm.extractvalue %3[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %20 = llvm.extractvalue %3[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %21 = llvm.extractvalue %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %22 = llvm.extractvalue %3[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %23 = llvm.call @qnode_forward_0.adjoint(%9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %8) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64) -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %24 = llvm.load %arg7 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64)>
    %25 = llvm.extractvalue %23[3] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %26 = llvm.alloca %0 x !llvm.array<1 x i64> : (i64) -> !llvm.ptr
    llvm.store %25, %26 : !llvm.array<1 x i64>, !llvm.ptr
    %27 = llvm.getelementptr inbounds %26[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x i64>
    %28 = llvm.load %27 : !llvm.ptr -> i64
    llvm.br ^bb1(%1 : i64)
  ^bb1(%29: i64):  // 2 preds: ^bb0, ^bb2
    %30 = llvm.icmp "slt" %29, %28 : i64
    llvm.cond_br %30, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %31 = llvm.extractvalue %24[1] : !llvm.struct<(ptr, ptr, i64)> 
    %32 = llvm.load %31 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    %33 = llvm.extractvalue %23[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %34 = llvm.getelementptr inbounds %33[%29] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %35 = llvm.load %34 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    %36 = llvm.extractvalue %4[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %37 = llvm.getelementptr inbounds %36[%29] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %38 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    %39 = llvm.fmul %32, %35 : f64
    %40 = llvm.fadd %38, %39 : f64
    %41 = llvm.extractvalue %4[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %42 = llvm.getelementptr inbounds %41[%29] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %40, %42 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %43 = llvm.add %29, %0 : i64
    llvm.br ^bb1(%43 : i64)
  ^bb3:  // pred: ^bb1
    llvm.return
  }
  llvm.func @qnode_forward_0.quantum(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr) attributes {gradient.qgrad = @qnode_forward_0.adjoint, passthrough = ["noinline"], sym_visibility = "private", unwrapped_type = (memref<4x8x3xf32>, memref<8xf32>, memref<?xf64>) -> memref<f64>} {
    %0 = llvm.mlir.constant(64 : index) : i64
    %1 = llvm.mlir.constant(2 : i64) : i64
    %2 = llvm.mlir.constant(3 : i64) : i64
    %3 = llvm.mlir.constant(4 : i64) : i64
    %4 = llvm.mlir.constant(5 : i64) : i64
    %5 = llvm.mlir.constant(6 : i64) : i64
    %6 = llvm.mlir.zero : !llvm.ptr
    %7 = llvm.mlir.constant(7 : i64) : i64
    %8 = llvm.mlir.constant(8 : i64) : i64
    %9 = llvm.mlir.constant(false) : i1
    %10 = llvm.mlir.addressof @"{}" : !llvm.ptr
    %11 = llvm.mlir.addressof @LightningGPUSimulator : !llvm.ptr
    %12 = llvm.mlir.addressof @"/home/pichau/QAgents/.venv/lib/python3.12/site-packages/pennylane_lightning/liblightning_gpu_catalyst.so" : !llvm.ptr
    %13 = llvm.mlir.constant(1 : index) : i64
    %14 = llvm.mlir.constant(1 : i64) : i64
    %15 = llvm.mlir.constant(0 : i64) : i64
    %16 = llvm.load volatile %arg0 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    llvm.store %16, %arg0 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>, !llvm.ptr
    %17 = llvm.load volatile %arg1 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    llvm.store %17, %arg1 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %18 = llvm.load volatile %arg2 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    llvm.store %18, %arg2 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %19 = llvm.load volatile %arg3 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64)>
    llvm.store %19, %arg3 : !llvm.struct<(ptr, ptr, i64)>, !llvm.ptr
    %20 = llvm.alloca %13 x i64 : (i64) -> !llvm.ptr
    llvm.store %15, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %21 = llvm.getelementptr inbounds %12[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<105 x i8>
    %22 = llvm.getelementptr inbounds %11[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<22 x i8>
    %23 = llvm.getelementptr inbounds %10[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3 x i8>
    llvm.call @__catalyst__rt__device_init(%21, %22, %23, %15, %9) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i1) -> ()
    %24 = llvm.call @__catalyst__rt__qubit_allocate_array(%8) : (i64) -> !llvm.ptr
    %25 = llvm.call @__catalyst__rt__array_get_element_ptr_1d(%24, %7) : (!llvm.ptr, i64) -> !llvm.ptr
    %26 = llvm.load %25 : !llvm.ptr -> !llvm.ptr
    %27 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %28 = llvm.add %27, %14 : i64
    llvm.store %28, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %29 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %30 = llvm.getelementptr inbounds %29[%27] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %31 = llvm.load %30 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RY(%31, %26, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %32 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %33 = llvm.add %32, %14 : i64
    llvm.store %33, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %34 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %35 = llvm.getelementptr inbounds %34[%32] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %36 = llvm.load %35 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RZ(%36, %26, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %37 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %38 = llvm.add %37, %14 : i64
    llvm.store %38, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %39 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %40 = llvm.getelementptr inbounds %39[%37] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %41 = llvm.load %40 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RY(%41, %26, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %42 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %43 = llvm.add %42, %14 : i64
    llvm.store %43, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %44 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %45 = llvm.getelementptr inbounds %44[%42] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %46 = llvm.load %45 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RZ(%46, %26, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %47 = llvm.call @__catalyst__rt__array_get_element_ptr_1d(%24, %5) : (!llvm.ptr, i64) -> !llvm.ptr
    %48 = llvm.load %47 : !llvm.ptr -> !llvm.ptr
    %49 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %50 = llvm.add %49, %14 : i64
    llvm.store %50, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %51 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %52 = llvm.getelementptr inbounds %51[%49] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %53 = llvm.load %52 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RY(%53, %48, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %54 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %55 = llvm.add %54, %14 : i64
    llvm.store %55, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %56 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %57 = llvm.getelementptr inbounds %56[%54] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %58 = llvm.load %57 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RZ(%58, %48, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %59 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %60 = llvm.add %59, %14 : i64
    llvm.store %60, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %61 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %62 = llvm.getelementptr inbounds %61[%59] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %63 = llvm.load %62 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RY(%63, %48, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %64 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %65 = llvm.add %64, %14 : i64
    llvm.store %65, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %66 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %67 = llvm.getelementptr inbounds %66[%64] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %68 = llvm.load %67 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RZ(%68, %48, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %69 = llvm.call @__catalyst__rt__array_get_element_ptr_1d(%24, %4) : (!llvm.ptr, i64) -> !llvm.ptr
    %70 = llvm.load %69 : !llvm.ptr -> !llvm.ptr
    %71 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %72 = llvm.add %71, %14 : i64
    llvm.store %72, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %73 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %74 = llvm.getelementptr inbounds %73[%71] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %75 = llvm.load %74 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RY(%75, %70, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %76 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %77 = llvm.add %76, %14 : i64
    llvm.store %77, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %78 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %79 = llvm.getelementptr inbounds %78[%76] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %80 = llvm.load %79 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RZ(%80, %70, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %81 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %82 = llvm.add %81, %14 : i64
    llvm.store %82, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %83 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %84 = llvm.getelementptr inbounds %83[%81] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %85 = llvm.load %84 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RY(%85, %70, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %86 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %87 = llvm.add %86, %14 : i64
    llvm.store %87, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %88 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %89 = llvm.getelementptr inbounds %88[%86] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %90 = llvm.load %89 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RZ(%90, %70, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %91 = llvm.call @__catalyst__rt__array_get_element_ptr_1d(%24, %3) : (!llvm.ptr, i64) -> !llvm.ptr
    %92 = llvm.load %91 : !llvm.ptr -> !llvm.ptr
    %93 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %94 = llvm.add %93, %14 : i64
    llvm.store %94, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %95 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %96 = llvm.getelementptr inbounds %95[%93] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %97 = llvm.load %96 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RY(%97, %92, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %98 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %99 = llvm.add %98, %14 : i64
    llvm.store %99, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %100 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %101 = llvm.getelementptr inbounds %100[%98] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %102 = llvm.load %101 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RZ(%102, %92, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %103 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %104 = llvm.add %103, %14 : i64
    llvm.store %104, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %105 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %106 = llvm.getelementptr inbounds %105[%103] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %107 = llvm.load %106 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RY(%107, %92, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %108 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %109 = llvm.add %108, %14 : i64
    llvm.store %109, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %110 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %111 = llvm.getelementptr inbounds %110[%108] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %112 = llvm.load %111 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RZ(%112, %92, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %113 = llvm.call @__catalyst__rt__array_get_element_ptr_1d(%24, %2) : (!llvm.ptr, i64) -> !llvm.ptr
    %114 = llvm.load %113 : !llvm.ptr -> !llvm.ptr
    %115 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %116 = llvm.add %115, %14 : i64
    llvm.store %116, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %117 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %118 = llvm.getelementptr inbounds %117[%115] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %119 = llvm.load %118 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RY(%119, %114, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %120 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %121 = llvm.add %120, %14 : i64
    llvm.store %121, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %122 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %123 = llvm.getelementptr inbounds %122[%120] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %124 = llvm.load %123 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RZ(%124, %114, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %125 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %126 = llvm.add %125, %14 : i64
    llvm.store %126, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %127 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %128 = llvm.getelementptr inbounds %127[%125] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %129 = llvm.load %128 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RY(%129, %114, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %130 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %131 = llvm.add %130, %14 : i64
    llvm.store %131, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %132 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %133 = llvm.getelementptr inbounds %132[%130] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %134 = llvm.load %133 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RZ(%134, %114, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %135 = llvm.call @__catalyst__rt__array_get_element_ptr_1d(%24, %1) : (!llvm.ptr, i64) -> !llvm.ptr
    %136 = llvm.load %135 : !llvm.ptr -> !llvm.ptr
    %137 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %138 = llvm.add %137, %14 : i64
    llvm.store %138, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %139 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %140 = llvm.getelementptr inbounds %139[%137] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %141 = llvm.load %140 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RY(%141, %136, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %142 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %143 = llvm.add %142, %14 : i64
    llvm.store %143, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %144 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %145 = llvm.getelementptr inbounds %144[%142] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %146 = llvm.load %145 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RZ(%146, %136, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %147 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %148 = llvm.add %147, %14 : i64
    llvm.store %148, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %149 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %150 = llvm.getelementptr inbounds %149[%147] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %151 = llvm.load %150 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RY(%151, %136, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %152 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %153 = llvm.add %152, %14 : i64
    llvm.store %153, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %154 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %155 = llvm.getelementptr inbounds %154[%152] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %156 = llvm.load %155 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RZ(%156, %136, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %157 = llvm.call @__catalyst__rt__array_get_element_ptr_1d(%24, %15) : (!llvm.ptr, i64) -> !llvm.ptr
    %158 = llvm.load %157 : !llvm.ptr -> !llvm.ptr
    %159 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %160 = llvm.add %159, %14 : i64
    llvm.store %160, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %161 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %162 = llvm.getelementptr inbounds %161[%159] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %163 = llvm.load %162 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RY(%163, %158, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %164 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %165 = llvm.add %164, %14 : i64
    llvm.store %165, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %166 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %167 = llvm.getelementptr inbounds %166[%164] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %168 = llvm.load %167 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RZ(%168, %158, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %169 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %170 = llvm.add %169, %14 : i64
    llvm.store %170, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %171 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %172 = llvm.getelementptr inbounds %171[%169] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %173 = llvm.load %172 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RY(%173, %158, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %174 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %175 = llvm.add %174, %14 : i64
    llvm.store %175, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %176 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %177 = llvm.getelementptr inbounds %176[%174] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %178 = llvm.load %177 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RZ(%178, %158, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %179 = llvm.call @__catalyst__rt__array_get_element_ptr_1d(%24, %14) : (!llvm.ptr, i64) -> !llvm.ptr
    %180 = llvm.load %179 : !llvm.ptr -> !llvm.ptr
    %181 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %182 = llvm.add %181, %14 : i64
    llvm.store %182, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %183 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %184 = llvm.getelementptr inbounds %183[%181] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %185 = llvm.load %184 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RY(%185, %180, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %186 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %187 = llvm.add %186, %14 : i64
    llvm.store %187, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %188 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %189 = llvm.getelementptr inbounds %188[%186] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %190 = llvm.load %189 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RZ(%190, %180, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %191 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %192 = llvm.add %191, %14 : i64
    llvm.store %192, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %193 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %194 = llvm.getelementptr inbounds %193[%191] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %195 = llvm.load %194 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RY(%195, %180, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %196 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %197 = llvm.add %196, %14 : i64
    llvm.store %197, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %198 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %199 = llvm.getelementptr inbounds %198[%196] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %200 = llvm.load %199 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RZ(%200, %180, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%158, %180, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%180, %136, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%136, %114, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%114, %92, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%92, %70, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%70, %48, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%48, %26, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %201 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %202 = llvm.add %201, %14 : i64
    llvm.store %202, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %203 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %204 = llvm.getelementptr inbounds %203[%201] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %205 = llvm.load %204 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RZ(%205, %48, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %206 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %207 = llvm.add %206, %14 : i64
    llvm.store %207, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %208 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %209 = llvm.getelementptr inbounds %208[%206] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %210 = llvm.load %209 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RY(%210, %48, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %211 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %212 = llvm.add %211, %14 : i64
    llvm.store %212, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %213 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %214 = llvm.getelementptr inbounds %213[%211] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %215 = llvm.load %214 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RZ(%215, %48, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %216 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %217 = llvm.add %216, %14 : i64
    llvm.store %217, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %218 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %219 = llvm.getelementptr inbounds %218[%216] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %220 = llvm.load %219 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RZ(%220, %92, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %221 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %222 = llvm.add %221, %14 : i64
    llvm.store %222, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %223 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %224 = llvm.getelementptr inbounds %223[%221] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %225 = llvm.load %224 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RY(%225, %92, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %226 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %227 = llvm.add %226, %14 : i64
    llvm.store %227, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %228 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %229 = llvm.getelementptr inbounds %228[%226] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %230 = llvm.load %229 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RZ(%230, %92, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%26, %158, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %231 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %232 = llvm.add %231, %14 : i64
    llvm.store %232, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %233 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %234 = llvm.getelementptr inbounds %233[%231] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %235 = llvm.load %234 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RZ(%235, %158, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %236 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %237 = llvm.add %236, %14 : i64
    llvm.store %237, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %238 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %239 = llvm.getelementptr inbounds %238[%236] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %240 = llvm.load %239 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RY(%240, %158, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %241 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %242 = llvm.add %241, %14 : i64
    llvm.store %242, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %243 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %244 = llvm.getelementptr inbounds %243[%241] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %245 = llvm.load %244 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RZ(%245, %158, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %246 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %247 = llvm.add %246, %14 : i64
    llvm.store %247, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %248 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %249 = llvm.getelementptr inbounds %248[%246] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %250 = llvm.load %249 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RZ(%250, %136, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %251 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %252 = llvm.add %251, %14 : i64
    llvm.store %252, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %253 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %254 = llvm.getelementptr inbounds %253[%251] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %255 = llvm.load %254 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RY(%255, %136, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %256 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %257 = llvm.add %256, %14 : i64
    llvm.store %257, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %258 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %259 = llvm.getelementptr inbounds %258[%256] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %260 = llvm.load %259 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RZ(%260, %136, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%158, %136, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%136, %92, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%92, %48, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%48, %158, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %261 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %262 = llvm.add %261, %14 : i64
    llvm.store %262, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %263 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %264 = llvm.getelementptr inbounds %263[%261] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %265 = llvm.load %264 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RZ(%265, %158, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %266 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %267 = llvm.add %266, %14 : i64
    llvm.store %267, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %268 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %269 = llvm.getelementptr inbounds %268[%266] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %270 = llvm.load %269 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RY(%270, %158, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %271 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %272 = llvm.add %271, %14 : i64
    llvm.store %272, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %273 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %274 = llvm.getelementptr inbounds %273[%271] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %275 = llvm.load %274 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RZ(%275, %158, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %276 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %277 = llvm.add %276, %14 : i64
    llvm.store %277, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %278 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %279 = llvm.getelementptr inbounds %278[%276] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %280 = llvm.load %279 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RZ(%280, %70, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %281 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %282 = llvm.add %281, %14 : i64
    llvm.store %282, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %283 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %284 = llvm.getelementptr inbounds %283[%281] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %285 = llvm.load %284 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RY(%285, %70, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %286 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %287 = llvm.add %286, %14 : i64
    llvm.store %287, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %288 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %289 = llvm.getelementptr inbounds %288[%286] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %290 = llvm.load %289 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RZ(%290, %70, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %291 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %292 = llvm.add %291, %14 : i64
    llvm.store %292, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %293 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %294 = llvm.getelementptr inbounds %293[%291] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %295 = llvm.load %294 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RZ(%295, %180, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %296 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %297 = llvm.add %296, %14 : i64
    llvm.store %297, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %298 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %299 = llvm.getelementptr inbounds %298[%296] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %300 = llvm.load %299 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RY(%300, %180, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %301 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %302 = llvm.add %301, %14 : i64
    llvm.store %302, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %303 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %304 = llvm.getelementptr inbounds %303[%301] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %305 = llvm.load %304 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RZ(%305, %180, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %306 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %307 = llvm.add %306, %14 : i64
    llvm.store %307, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %308 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %309 = llvm.getelementptr inbounds %308[%306] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %310 = llvm.load %309 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RZ(%310, %114, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %311 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %312 = llvm.add %311, %14 : i64
    llvm.store %312, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %313 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %314 = llvm.getelementptr inbounds %313[%311] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %315 = llvm.load %314 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RY(%315, %114, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %316 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %317 = llvm.add %316, %14 : i64
    llvm.store %317, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %318 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %319 = llvm.getelementptr inbounds %318[%316] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %320 = llvm.load %319 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RZ(%320, %114, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%180, %114, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%114, %70, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %321 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %322 = llvm.add %321, %14 : i64
    llvm.store %322, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %323 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %324 = llvm.getelementptr inbounds %323[%321] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %325 = llvm.load %324 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RZ(%325, %114, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %326 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %327 = llvm.add %326, %14 : i64
    llvm.store %327, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %328 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %329 = llvm.getelementptr inbounds %328[%326] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %330 = llvm.load %329 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RY(%330, %114, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %331 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %332 = llvm.add %331, %14 : i64
    llvm.store %332, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %333 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %334 = llvm.getelementptr inbounds %333[%331] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %335 = llvm.load %334 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RZ(%335, %114, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%158, %114, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %336 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %337 = llvm.add %336, %14 : i64
    llvm.store %337, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %338 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %339 = llvm.getelementptr inbounds %338[%336] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %340 = llvm.load %339 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RZ(%340, %136, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %341 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %342 = llvm.add %341, %14 : i64
    llvm.store %342, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %343 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %344 = llvm.getelementptr inbounds %343[%341] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %345 = llvm.load %344 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RY(%345, %136, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %346 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %347 = llvm.add %346, %14 : i64
    llvm.store %347, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %348 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %349 = llvm.getelementptr inbounds %348[%346] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %350 = llvm.load %349 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RZ(%350, %136, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %351 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %352 = llvm.add %351, %14 : i64
    llvm.store %352, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %353 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %354 = llvm.getelementptr inbounds %353[%351] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %355 = llvm.load %354 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RZ(%355, %26, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %356 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %357 = llvm.add %356, %14 : i64
    llvm.store %357, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %358 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %359 = llvm.getelementptr inbounds %358[%356] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %360 = llvm.load %359 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RY(%360, %26, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %361 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %362 = llvm.add %361, %14 : i64
    llvm.store %362, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %363 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %364 = llvm.getelementptr inbounds %363[%361] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %365 = llvm.load %364 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RZ(%365, %26, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%70, %26, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %366 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %367 = llvm.add %366, %14 : i64
    llvm.store %367, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %368 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %369 = llvm.getelementptr inbounds %368[%366] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %370 = llvm.load %369 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RZ(%370, %70, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %371 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %372 = llvm.add %371, %14 : i64
    llvm.store %372, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %373 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %374 = llvm.getelementptr inbounds %373[%371] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %375 = llvm.load %374 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RY(%375, %70, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %376 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %377 = llvm.add %376, %14 : i64
    llvm.store %377, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %378 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %379 = llvm.getelementptr inbounds %378[%376] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %380 = llvm.load %379 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RZ(%380, %70, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%136, %70, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%70, %158, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %381 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %382 = llvm.add %381, %14 : i64
    llvm.store %382, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %383 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %384 = llvm.getelementptr inbounds %383[%381] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %385 = llvm.load %384 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RZ(%385, %158, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %386 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %387 = llvm.add %386, %14 : i64
    llvm.store %387, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %388 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %389 = llvm.getelementptr inbounds %388[%386] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %390 = llvm.load %389 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RY(%390, %158, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %391 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %392 = llvm.add %391, %14 : i64
    llvm.store %392, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %393 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %394 = llvm.getelementptr inbounds %393[%391] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %395 = llvm.load %394 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RZ(%395, %158, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%26, %180, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %396 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %397 = llvm.add %396, %14 : i64
    llvm.store %397, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %398 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %399 = llvm.getelementptr inbounds %398[%396] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %400 = llvm.load %399 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RZ(%400, %26, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %401 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %402 = llvm.add %401, %14 : i64
    llvm.store %402, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %403 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %404 = llvm.getelementptr inbounds %403[%401] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %405 = llvm.load %404 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RY(%405, %26, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %406 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %407 = llvm.add %406, %14 : i64
    llvm.store %407, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %408 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %409 = llvm.getelementptr inbounds %408[%406] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %410 = llvm.load %409 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RZ(%410, %26, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %411 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %412 = llvm.add %411, %14 : i64
    llvm.store %412, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %413 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %414 = llvm.getelementptr inbounds %413[%411] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %415 = llvm.load %414 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RZ(%415, %180, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %416 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %417 = llvm.add %416, %14 : i64
    llvm.store %417, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %418 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %419 = llvm.getelementptr inbounds %418[%416] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %420 = llvm.load %419 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RY(%420, %180, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %421 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %422 = llvm.add %421, %14 : i64
    llvm.store %422, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %423 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %424 = llvm.getelementptr inbounds %423[%421] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %425 = llvm.load %424 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RZ(%425, %180, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %426 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %427 = llvm.add %426, %14 : i64
    llvm.store %427, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %428 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %429 = llvm.getelementptr inbounds %428[%426] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %430 = llvm.load %429 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RZ(%430, %92, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %431 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %432 = llvm.add %431, %14 : i64
    llvm.store %432, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %433 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %434 = llvm.getelementptr inbounds %433[%431] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %435 = llvm.load %434 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RY(%435, %92, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %436 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %437 = llvm.add %436, %14 : i64
    llvm.store %437, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %438 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %439 = llvm.getelementptr inbounds %438[%436] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %440 = llvm.load %439 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RZ(%440, %92, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%180, %92, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%92, %26, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %441 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %442 = llvm.add %441, %14 : i64
    llvm.store %442, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %443 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %444 = llvm.getelementptr inbounds %443[%441] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %445 = llvm.load %444 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RZ(%445, %92, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %446 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %447 = llvm.add %446, %14 : i64
    llvm.store %447, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %448 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %449 = llvm.getelementptr inbounds %448[%446] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %450 = llvm.load %449 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RY(%450, %92, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %451 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %452 = llvm.add %451, %14 : i64
    llvm.store %452, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %453 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %454 = llvm.getelementptr inbounds %453[%451] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %455 = llvm.load %454 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RZ(%455, %92, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%158, %92, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%92, %158, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %456 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %457 = llvm.add %456, %14 : i64
    llvm.store %457, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %458 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %459 = llvm.getelementptr inbounds %458[%456] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %460 = llvm.load %459 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RZ(%460, %48, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %461 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %462 = llvm.add %461, %14 : i64
    llvm.store %462, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %463 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %464 = llvm.getelementptr inbounds %463[%461] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %465 = llvm.load %464 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RY(%465, %48, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %466 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %467 = llvm.add %466, %14 : i64
    llvm.store %467, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %468 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %469 = llvm.getelementptr inbounds %468[%466] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %470 = llvm.load %469 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RZ(%470, %48, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%114, %48, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%48, %180, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %471 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %472 = llvm.add %471, %14 : i64
    llvm.store %472, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %473 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %474 = llvm.getelementptr inbounds %473[%471] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %475 = llvm.load %474 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RZ(%475, %180, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %476 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %477 = llvm.add %476, %14 : i64
    llvm.store %477, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %478 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %479 = llvm.getelementptr inbounds %478[%476] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %480 = llvm.load %479 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RY(%480, %180, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %481 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %482 = llvm.add %481, %14 : i64
    llvm.store %482, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %483 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %484 = llvm.getelementptr inbounds %483[%481] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %485 = llvm.load %484 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RZ(%485, %180, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %486 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %487 = llvm.add %486, %14 : i64
    llvm.store %487, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %488 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %489 = llvm.getelementptr inbounds %488[%486] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %490 = llvm.load %489 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RZ(%490, %70, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %491 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %492 = llvm.add %491, %14 : i64
    llvm.store %492, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %493 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %494 = llvm.getelementptr inbounds %493[%491] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %495 = llvm.load %494 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RY(%495, %70, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %496 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %497 = llvm.add %496, %14 : i64
    llvm.store %497, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %498 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %499 = llvm.getelementptr inbounds %498[%496] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %500 = llvm.load %499 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RZ(%500, %70, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%180, %70, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%70, %180, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%26, %136, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %501 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %502 = llvm.add %501, %14 : i64
    llvm.store %502, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %503 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %504 = llvm.getelementptr inbounds %503[%501] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %505 = llvm.load %504 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RZ(%505, %136, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %506 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %507 = llvm.add %506, %14 : i64
    llvm.store %507, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %508 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %509 = llvm.getelementptr inbounds %508[%506] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %510 = llvm.load %509 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RY(%510, %136, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %511 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %512 = llvm.add %511, %14 : i64
    llvm.store %512, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %513 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %514 = llvm.getelementptr inbounds %513[%511] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %515 = llvm.load %514 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RZ(%515, %136, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %516 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %517 = llvm.add %516, %14 : i64
    llvm.store %517, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %518 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %519 = llvm.getelementptr inbounds %518[%516] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %520 = llvm.load %519 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RZ(%520, %48, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %521 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %522 = llvm.add %521, %14 : i64
    llvm.store %522, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %523 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %524 = llvm.getelementptr inbounds %523[%521] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %525 = llvm.load %524 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RY(%525, %48, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %526 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %527 = llvm.add %526, %14 : i64
    llvm.store %527, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %528 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %529 = llvm.getelementptr inbounds %528[%526] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %530 = llvm.load %529 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RZ(%530, %48, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%136, %48, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%48, %136, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %531 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %532 = llvm.add %531, %14 : i64
    llvm.store %532, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %533 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %534 = llvm.getelementptr inbounds %533[%531] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %535 = llvm.load %534 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RZ(%535, %114, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %536 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %537 = llvm.add %536, %14 : i64
    llvm.store %537, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %538 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %539 = llvm.getelementptr inbounds %538[%536] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %540 = llvm.load %539 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RY(%540, %114, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %541 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %542 = llvm.add %541, %14 : i64
    llvm.store %542, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %543 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %544 = llvm.getelementptr inbounds %543[%541] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %545 = llvm.load %544 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RZ(%545, %114, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %546 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %547 = llvm.add %546, %14 : i64
    llvm.store %547, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %548 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %549 = llvm.getelementptr inbounds %548[%546] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %550 = llvm.load %549 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RZ(%550, %26, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %551 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %552 = llvm.add %551, %14 : i64
    llvm.store %552, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %553 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %554 = llvm.getelementptr inbounds %553[%551] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %555 = llvm.load %554 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RY(%555, %26, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %556 = llvm.load %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %557 = llvm.add %556, %14 : i64
    llvm.store %557, %20 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %558 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %559 = llvm.getelementptr inbounds %558[%556] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %560 = llvm.load %559 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RZ(%560, %26, %6) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%114, %26, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%26, %114, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %561 = llvm.call @__catalyst__qis__NamedObs(%2, %158) : (i64, !llvm.ptr) -> i64
    %562 = llvm.call @__catalyst__qis__Expval(%561) : (i64) -> f64
    %563 = llvm.getelementptr %6[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %564 = llvm.ptrtoint %563 : !llvm.ptr to i64
    %565 = llvm.add %564, %0 : i64
    %566 = llvm.call @_mlir_memref_to_llvm_alloc(%565) : (i64) -> !llvm.ptr
    %567 = llvm.ptrtoint %566 : !llvm.ptr to i64
    %568 = llvm.sub %0, %13 : i64
    %569 = llvm.add %567, %568 : i64
    %570 = llvm.urem %569, %0 : i64
    %571 = llvm.sub %569, %570 : i64
    %572 = llvm.inttoptr %571 : i64 to !llvm.ptr
    llvm.store %562, %572 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    llvm.call @__catalyst__rt__qubit_release_array(%24) : (!llvm.ptr) -> ()
    llvm.call @__catalyst__rt__device_release() : () -> ()
    %573 = llvm.load %572 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    %574 = llvm.extractvalue %19[1] : !llvm.struct<(ptr, ptr, i64)> 
    llvm.store %573, %574 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    llvm.return
  }
  llvm.func @qnode_forward_0.quantum.augfwd(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: !llvm.ptr, %arg6: !llvm.ptr, %arg7: !llvm.ptr) -> !llvm.ptr attributes {sym_visibility = "private"} {
    %0 = llvm.mlir.zero : !llvm.ptr
    llvm.call @qnode_forward_0.quantum(%arg0, %arg2, %arg4, %arg6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.return %0 : !llvm.ptr
  }
  llvm.func @qnode_forward_0.preprocess(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: !llvm.ptr, %arg10: !llvm.ptr, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: i64, %arg15: !llvm.ptr, %arg16: !llvm.ptr, %arg17: i64) attributes {sym_visibility = "private"} {
    %0 = llvm.mlir.constant(64 : index) : i64
    %1 = llvm.mlir.addressof @__constant_xf32 : !llvm.ptr
    %2 = llvm.mlir.zero : !llvm.ptr
    %3 = llvm.mlir.constant(0 : i64) : i64
    %4 = llvm.mlir.constant(1 : i64) : i64
    %5 = llvm.mlir.constant(0 : index) : i64
    %6 = llvm.mlir.constant(8 : index) : i64
    %7 = llvm.mlir.constant(1 : index) : i64
    %8 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %9 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %10 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64)>
    %11 = llvm.insertvalue %arg9, %9[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %12 = llvm.insertvalue %arg10, %11[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %13 = llvm.insertvalue %arg11, %12[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %14 = llvm.insertvalue %arg12, %13[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %15 = llvm.insertvalue %arg13, %14[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %16 = llvm.insertvalue %arg0, %8[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %17 = llvm.insertvalue %arg1, %16[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %18 = llvm.insertvalue %arg2, %17[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %19 = llvm.insertvalue %arg3, %18[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %20 = llvm.insertvalue %arg6, %19[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %21 = llvm.insertvalue %arg4, %20[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %22 = llvm.insertvalue %arg7, %21[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %23 = llvm.insertvalue %arg5, %22[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %24 = llvm.insertvalue %arg8, %23[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %25 = llvm.alloca %4 x !llvm.struct<(ptr, ptr, i64)> : (i64) -> !llvm.ptr
    %26 = llvm.alloca %4 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    %27 = llvm.alloca %4 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    %28 = llvm.alloca %4 x !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> : (i64) -> !llvm.ptr
    %29 = llvm.getelementptr %2[%arg14] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %30 = llvm.ptrtoint %29 : !llvm.ptr to i64
    %31 = llvm.call @_mlir_memref_to_llvm_alloc(%30) : (i64) -> !llvm.ptr
    %32 = llvm.insertvalue %31, %9[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %33 = llvm.insertvalue %31, %32[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %34 = llvm.insertvalue %5, %33[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %35 = llvm.insertvalue %arg14, %34[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %36 = llvm.insertvalue %7, %35[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %37 = llvm.alloca %7 x i64 : (i64) -> !llvm.ptr
    llvm.store %3, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %38 = llvm.getelementptr inbounds %arg1[74] : (!llvm.ptr) -> !llvm.ptr, f32
    %39 = llvm.load %38 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %40 = llvm.getelementptr inbounds %arg1[73] : (!llvm.ptr) -> !llvm.ptr, f32
    %41 = llvm.load %40 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %42 = llvm.getelementptr inbounds %arg1[72] : (!llvm.ptr) -> !llvm.ptr, f32
    %43 = llvm.load %42 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %44 = llvm.getelementptr inbounds %arg1[50] : (!llvm.ptr) -> !llvm.ptr, f32
    %45 = llvm.load %44 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %46 = llvm.getelementptr inbounds %arg1[49] : (!llvm.ptr) -> !llvm.ptr, f32
    %47 = llvm.load %46 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %48 = llvm.getelementptr inbounds %arg1[48] : (!llvm.ptr) -> !llvm.ptr, f32
    %49 = llvm.load %48 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %50 = llvm.getelementptr inbounds %arg1[44] : (!llvm.ptr) -> !llvm.ptr, f32
    %51 = llvm.load %50 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %52 = llvm.getelementptr inbounds %arg1[43] : (!llvm.ptr) -> !llvm.ptr, f32
    %53 = llvm.load %52 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %54 = llvm.getelementptr inbounds %arg1[42] : (!llvm.ptr) -> !llvm.ptr, f32
    %55 = llvm.load %54 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %56 = llvm.getelementptr inbounds %arg1[23] : (!llvm.ptr) -> !llvm.ptr, f32
    %57 = llvm.load %56 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %58 = llvm.getelementptr inbounds %arg1[22] : (!llvm.ptr) -> !llvm.ptr, f32
    %59 = llvm.load %58 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %60 = llvm.getelementptr inbounds %arg1[21] : (!llvm.ptr) -> !llvm.ptr, f32
    %61 = llvm.load %60 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %62 = llvm.getelementptr %2[8] : (!llvm.ptr) -> !llvm.ptr, f32
    %63 = llvm.ptrtoint %62 : !llvm.ptr to i64
    %64 = llvm.add %63, %0 : i64
    %65 = llvm.call @_mlir_memref_to_llvm_alloc(%64) : (i64) -> !llvm.ptr
    %66 = llvm.ptrtoint %65 : !llvm.ptr to i64
    %67 = llvm.sub %0, %7 : i64
    %68 = llvm.add %66, %67 : i64
    %69 = llvm.urem %68, %0 : i64
    %70 = llvm.sub %68, %69 : i64
    %71 = llvm.inttoptr %70 : i64 to !llvm.ptr
    llvm.br ^bb1(%5 : i64)
  ^bb1(%72: i64):  // 2 preds: ^bb0, ^bb2
    %73 = llvm.icmp "slt" %72, %6 : i64
    llvm.cond_br %73, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %74 = llvm.load %1 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %75 = llvm.getelementptr inbounds %71[%72] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %74, %75 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f32, !llvm.ptr
    %76 = llvm.add %72, %7 : i64
    llvm.br ^bb1(%76 : i64)
  ^bb3:  // pred: ^bb1
    llvm.br ^bb4(%5 : i64)
  ^bb4(%77: i64):  // 2 preds: ^bb3, ^bb5
    %78 = llvm.icmp "slt" %77, %6 : i64
    llvm.cond_br %78, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %79 = llvm.getelementptr inbounds %71[%77] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %80 = llvm.load %79 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %81 = llvm.getelementptr inbounds %arg10[%77] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %82 = llvm.load %81 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %83 = llvm.fmul %80, %82 : f32
    %84 = llvm.getelementptr inbounds %71[%77] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %83, %84 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f32, !llvm.ptr
    %85 = llvm.add %77, %7 : i64
    llvm.br ^bb4(%85 : i64)
  ^bb6:  // pred: ^bb4
    %86 = llvm.getelementptr inbounds %71[7] : (!llvm.ptr) -> !llvm.ptr, f32
    %87 = llvm.load %86 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %88 = llvm.fpext %87 : f32 to f64
    %89 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %90 = llvm.getelementptr inbounds %31[%89] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %88, %90 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %91 = llvm.add %89, %4 : i64
    llvm.store %91, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %92 = llvm.fpext %61 : f32 to f64
    %93 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %94 = llvm.getelementptr inbounds %31[%93] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %92, %94 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %95 = llvm.add %93, %4 : i64
    llvm.store %95, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %96 = llvm.fpext %59 : f32 to f64
    %97 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %98 = llvm.getelementptr inbounds %31[%97] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %96, %98 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %99 = llvm.add %97, %4 : i64
    llvm.store %99, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %100 = llvm.fpext %57 : f32 to f64
    %101 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %102 = llvm.getelementptr inbounds %31[%101] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %100, %102 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %103 = llvm.add %101, %4 : i64
    llvm.store %103, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %104 = llvm.getelementptr inbounds %arg1[20] : (!llvm.ptr) -> !llvm.ptr, f32
    %105 = llvm.load %104 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %106 = llvm.getelementptr inbounds %arg1[19] : (!llvm.ptr) -> !llvm.ptr, f32
    %107 = llvm.load %106 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %108 = llvm.getelementptr inbounds %arg1[18] : (!llvm.ptr) -> !llvm.ptr, f32
    %109 = llvm.load %108 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %110 = llvm.getelementptr inbounds %71[6] : (!llvm.ptr) -> !llvm.ptr, f32
    %111 = llvm.load %110 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %112 = llvm.fpext %111 : f32 to f64
    %113 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %114 = llvm.getelementptr inbounds %31[%113] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %112, %114 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %115 = llvm.add %113, %4 : i64
    llvm.store %115, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %116 = llvm.fpext %109 : f32 to f64
    %117 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %118 = llvm.getelementptr inbounds %31[%117] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %116, %118 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %119 = llvm.add %117, %4 : i64
    llvm.store %119, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %120 = llvm.fpext %107 : f32 to f64
    %121 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %122 = llvm.getelementptr inbounds %31[%121] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %120, %122 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %123 = llvm.add %121, %4 : i64
    llvm.store %123, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %124 = llvm.fpext %105 : f32 to f64
    %125 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %126 = llvm.getelementptr inbounds %31[%125] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %124, %126 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %127 = llvm.add %125, %4 : i64
    llvm.store %127, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %128 = llvm.getelementptr inbounds %arg1[17] : (!llvm.ptr) -> !llvm.ptr, f32
    %129 = llvm.load %128 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %130 = llvm.getelementptr inbounds %arg1[16] : (!llvm.ptr) -> !llvm.ptr, f32
    %131 = llvm.load %130 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %132 = llvm.getelementptr inbounds %arg1[15] : (!llvm.ptr) -> !llvm.ptr, f32
    %133 = llvm.load %132 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %134 = llvm.getelementptr inbounds %71[5] : (!llvm.ptr) -> !llvm.ptr, f32
    %135 = llvm.load %134 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %136 = llvm.fpext %135 : f32 to f64
    %137 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %138 = llvm.getelementptr inbounds %31[%137] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %136, %138 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %139 = llvm.add %137, %4 : i64
    llvm.store %139, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %140 = llvm.fpext %133 : f32 to f64
    %141 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %142 = llvm.getelementptr inbounds %31[%141] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %140, %142 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %143 = llvm.add %141, %4 : i64
    llvm.store %143, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %144 = llvm.fpext %131 : f32 to f64
    %145 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %146 = llvm.getelementptr inbounds %31[%145] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %144, %146 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %147 = llvm.add %145, %4 : i64
    llvm.store %147, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %148 = llvm.fpext %129 : f32 to f64
    %149 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %150 = llvm.getelementptr inbounds %31[%149] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %148, %150 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %151 = llvm.add %149, %4 : i64
    llvm.store %151, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %152 = llvm.getelementptr inbounds %arg1[14] : (!llvm.ptr) -> !llvm.ptr, f32
    %153 = llvm.load %152 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %154 = llvm.getelementptr inbounds %arg1[13] : (!llvm.ptr) -> !llvm.ptr, f32
    %155 = llvm.load %154 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %156 = llvm.getelementptr inbounds %arg1[12] : (!llvm.ptr) -> !llvm.ptr, f32
    %157 = llvm.load %156 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %158 = llvm.getelementptr inbounds %71[4] : (!llvm.ptr) -> !llvm.ptr, f32
    %159 = llvm.load %158 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %160 = llvm.fpext %159 : f32 to f64
    %161 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %162 = llvm.getelementptr inbounds %31[%161] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %160, %162 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %163 = llvm.add %161, %4 : i64
    llvm.store %163, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %164 = llvm.fpext %157 : f32 to f64
    %165 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %166 = llvm.getelementptr inbounds %31[%165] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %164, %166 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %167 = llvm.add %165, %4 : i64
    llvm.store %167, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %168 = llvm.fpext %155 : f32 to f64
    %169 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %170 = llvm.getelementptr inbounds %31[%169] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %168, %170 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %171 = llvm.add %169, %4 : i64
    llvm.store %171, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %172 = llvm.fpext %153 : f32 to f64
    %173 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %174 = llvm.getelementptr inbounds %31[%173] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %172, %174 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %175 = llvm.add %173, %4 : i64
    llvm.store %175, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %176 = llvm.getelementptr inbounds %arg1[11] : (!llvm.ptr) -> !llvm.ptr, f32
    %177 = llvm.load %176 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %178 = llvm.getelementptr inbounds %arg1[10] : (!llvm.ptr) -> !llvm.ptr, f32
    %179 = llvm.load %178 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %180 = llvm.getelementptr inbounds %arg1[9] : (!llvm.ptr) -> !llvm.ptr, f32
    %181 = llvm.load %180 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %182 = llvm.getelementptr inbounds %71[3] : (!llvm.ptr) -> !llvm.ptr, f32
    %183 = llvm.load %182 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %184 = llvm.fpext %183 : f32 to f64
    %185 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %186 = llvm.getelementptr inbounds %31[%185] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %184, %186 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %187 = llvm.add %185, %4 : i64
    llvm.store %187, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %188 = llvm.fpext %181 : f32 to f64
    %189 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %190 = llvm.getelementptr inbounds %31[%189] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %188, %190 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %191 = llvm.add %189, %4 : i64
    llvm.store %191, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %192 = llvm.fpext %179 : f32 to f64
    %193 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %194 = llvm.getelementptr inbounds %31[%193] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %192, %194 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %195 = llvm.add %193, %4 : i64
    llvm.store %195, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %196 = llvm.fpext %177 : f32 to f64
    %197 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %198 = llvm.getelementptr inbounds %31[%197] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %196, %198 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %199 = llvm.add %197, %4 : i64
    llvm.store %199, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %200 = llvm.getelementptr inbounds %arg1[8] : (!llvm.ptr) -> !llvm.ptr, f32
    %201 = llvm.load %200 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %202 = llvm.getelementptr inbounds %arg1[7] : (!llvm.ptr) -> !llvm.ptr, f32
    %203 = llvm.load %202 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %204 = llvm.getelementptr inbounds %arg1[6] : (!llvm.ptr) -> !llvm.ptr, f32
    %205 = llvm.load %204 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %206 = llvm.getelementptr inbounds %71[2] : (!llvm.ptr) -> !llvm.ptr, f32
    %207 = llvm.load %206 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %208 = llvm.fpext %207 : f32 to f64
    %209 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %210 = llvm.getelementptr inbounds %31[%209] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %208, %210 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %211 = llvm.add %209, %4 : i64
    llvm.store %211, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %212 = llvm.fpext %205 : f32 to f64
    %213 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %214 = llvm.getelementptr inbounds %31[%213] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %212, %214 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %215 = llvm.add %213, %4 : i64
    llvm.store %215, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %216 = llvm.fpext %203 : f32 to f64
    %217 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %218 = llvm.getelementptr inbounds %31[%217] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %216, %218 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %219 = llvm.add %217, %4 : i64
    llvm.store %219, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %220 = llvm.fpext %201 : f32 to f64
    %221 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %222 = llvm.getelementptr inbounds %31[%221] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %220, %222 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %223 = llvm.add %221, %4 : i64
    llvm.store %223, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %224 = llvm.getelementptr inbounds %arg1[2] : (!llvm.ptr) -> !llvm.ptr, f32
    %225 = llvm.load %224 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %226 = llvm.getelementptr inbounds %arg1[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %227 = llvm.load %226 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %228 = llvm.load %arg1 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %229 = llvm.load %71 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %230 = llvm.fpext %229 : f32 to f64
    %231 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %232 = llvm.getelementptr inbounds %31[%231] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %230, %232 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %233 = llvm.add %231, %4 : i64
    llvm.store %233, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %234 = llvm.fpext %228 : f32 to f64
    %235 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %236 = llvm.getelementptr inbounds %31[%235] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %234, %236 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %237 = llvm.add %235, %4 : i64
    llvm.store %237, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %238 = llvm.fpext %227 : f32 to f64
    %239 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %240 = llvm.getelementptr inbounds %31[%239] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %238, %240 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %241 = llvm.add %239, %4 : i64
    llvm.store %241, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %242 = llvm.fpext %225 : f32 to f64
    %243 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %244 = llvm.getelementptr inbounds %31[%243] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %242, %244 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %245 = llvm.add %243, %4 : i64
    llvm.store %245, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %246 = llvm.getelementptr inbounds %arg1[5] : (!llvm.ptr) -> !llvm.ptr, f32
    %247 = llvm.load %246 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %248 = llvm.getelementptr inbounds %arg1[4] : (!llvm.ptr) -> !llvm.ptr, f32
    %249 = llvm.load %248 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %250 = llvm.getelementptr inbounds %arg1[3] : (!llvm.ptr) -> !llvm.ptr, f32
    %251 = llvm.load %250 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %252 = llvm.getelementptr inbounds %71[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %253 = llvm.load %252 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    llvm.call @_mlir_memref_to_llvm_free(%65) : (!llvm.ptr) -> ()
    %254 = llvm.fpext %253 : f32 to f64
    %255 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %256 = llvm.getelementptr inbounds %31[%255] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %254, %256 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %257 = llvm.add %255, %4 : i64
    llvm.store %257, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %258 = llvm.fpext %251 : f32 to f64
    %259 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %260 = llvm.getelementptr inbounds %31[%259] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %258, %260 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %261 = llvm.add %259, %4 : i64
    llvm.store %261, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %262 = llvm.fpext %249 : f32 to f64
    %263 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %264 = llvm.getelementptr inbounds %31[%263] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %262, %264 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %265 = llvm.add %263, %4 : i64
    llvm.store %265, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %266 = llvm.fpext %247 : f32 to f64
    %267 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %268 = llvm.getelementptr inbounds %31[%267] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %266, %268 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %269 = llvm.add %267, %4 : i64
    llvm.store %269, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %270 = llvm.fpext %55 : f32 to f64
    %271 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %272 = llvm.getelementptr inbounds %31[%271] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %270, %272 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %273 = llvm.add %271, %4 : i64
    llvm.store %273, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %274 = llvm.fpext %53 : f32 to f64
    %275 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %276 = llvm.getelementptr inbounds %31[%275] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %274, %276 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %277 = llvm.add %275, %4 : i64
    llvm.store %277, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %278 = llvm.fpext %51 : f32 to f64
    %279 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %280 = llvm.getelementptr inbounds %31[%279] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %278, %280 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %281 = llvm.add %279, %4 : i64
    llvm.store %281, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %282 = llvm.getelementptr inbounds %arg1[38] : (!llvm.ptr) -> !llvm.ptr, f32
    %283 = llvm.load %282 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %284 = llvm.getelementptr inbounds %arg1[37] : (!llvm.ptr) -> !llvm.ptr, f32
    %285 = llvm.load %284 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %286 = llvm.getelementptr inbounds %arg1[36] : (!llvm.ptr) -> !llvm.ptr, f32
    %287 = llvm.load %286 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %288 = llvm.fpext %287 : f32 to f64
    %289 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %290 = llvm.getelementptr inbounds %31[%289] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %288, %290 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %291 = llvm.add %289, %4 : i64
    llvm.store %291, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %292 = llvm.fpext %285 : f32 to f64
    %293 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %294 = llvm.getelementptr inbounds %31[%293] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %292, %294 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %295 = llvm.add %293, %4 : i64
    llvm.store %295, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %296 = llvm.fpext %283 : f32 to f64
    %297 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %298 = llvm.getelementptr inbounds %31[%297] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %296, %298 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %299 = llvm.add %297, %4 : i64
    llvm.store %299, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %300 = llvm.getelementptr inbounds %arg1[26] : (!llvm.ptr) -> !llvm.ptr, f32
    %301 = llvm.load %300 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %302 = llvm.getelementptr inbounds %arg1[25] : (!llvm.ptr) -> !llvm.ptr, f32
    %303 = llvm.load %302 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %304 = llvm.getelementptr inbounds %arg1[24] : (!llvm.ptr) -> !llvm.ptr, f32
    %305 = llvm.load %304 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %306 = llvm.fpext %305 : f32 to f64
    %307 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %308 = llvm.getelementptr inbounds %31[%307] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %306, %308 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %309 = llvm.add %307, %4 : i64
    llvm.store %309, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %310 = llvm.fpext %303 : f32 to f64
    %311 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %312 = llvm.getelementptr inbounds %31[%311] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %310, %312 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %313 = llvm.add %311, %4 : i64
    llvm.store %313, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %314 = llvm.fpext %301 : f32 to f64
    %315 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %316 = llvm.getelementptr inbounds %31[%315] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %314, %316 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %317 = llvm.add %315, %4 : i64
    llvm.store %317, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %318 = llvm.getelementptr inbounds %arg1[32] : (!llvm.ptr) -> !llvm.ptr, f32
    %319 = llvm.load %318 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %320 = llvm.getelementptr inbounds %arg1[31] : (!llvm.ptr) -> !llvm.ptr, f32
    %321 = llvm.load %320 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %322 = llvm.getelementptr inbounds %arg1[30] : (!llvm.ptr) -> !llvm.ptr, f32
    %323 = llvm.load %322 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %324 = llvm.fpext %323 : f32 to f64
    %325 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %326 = llvm.getelementptr inbounds %31[%325] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %324, %326 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %327 = llvm.add %325, %4 : i64
    llvm.store %327, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %328 = llvm.fpext %321 : f32 to f64
    %329 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %330 = llvm.getelementptr inbounds %31[%329] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %328, %330 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %331 = llvm.add %329, %4 : i64
    llvm.store %331, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %332 = llvm.fpext %319 : f32 to f64
    %333 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %334 = llvm.getelementptr inbounds %31[%333] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %332, %334 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %335 = llvm.add %333, %4 : i64
    llvm.store %335, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %336 = llvm.fpext %49 : f32 to f64
    %337 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %338 = llvm.getelementptr inbounds %31[%337] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %336, %338 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %339 = llvm.add %337, %4 : i64
    llvm.store %339, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %340 = llvm.fpext %47 : f32 to f64
    %341 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %342 = llvm.getelementptr inbounds %31[%341] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %340, %342 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %343 = llvm.add %341, %4 : i64
    llvm.store %343, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %344 = llvm.fpext %45 : f32 to f64
    %345 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %346 = llvm.getelementptr inbounds %31[%345] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %344, %346 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %347 = llvm.add %345, %4 : i64
    llvm.store %347, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %348 = llvm.getelementptr inbounds %arg1[59] : (!llvm.ptr) -> !llvm.ptr, f32
    %349 = llvm.load %348 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %350 = llvm.getelementptr inbounds %arg1[58] : (!llvm.ptr) -> !llvm.ptr, f32
    %351 = llvm.load %350 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %352 = llvm.getelementptr inbounds %arg1[57] : (!llvm.ptr) -> !llvm.ptr, f32
    %353 = llvm.load %352 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %354 = llvm.getelementptr inbounds %arg1[41] : (!llvm.ptr) -> !llvm.ptr, f32
    %355 = llvm.load %354 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %356 = llvm.getelementptr inbounds %arg1[40] : (!llvm.ptr) -> !llvm.ptr, f32
    %357 = llvm.load %356 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %358 = llvm.getelementptr inbounds %arg1[39] : (!llvm.ptr) -> !llvm.ptr, f32
    %359 = llvm.load %358 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %360 = llvm.fpext %359 : f32 to f64
    %361 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %362 = llvm.getelementptr inbounds %31[%361] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %360, %362 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %363 = llvm.add %361, %4 : i64
    llvm.store %363, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %364 = llvm.fpext %357 : f32 to f64
    %365 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %366 = llvm.getelementptr inbounds %31[%365] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %364, %366 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %367 = llvm.add %365, %4 : i64
    llvm.store %367, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %368 = llvm.fpext %355 : f32 to f64
    %369 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %370 = llvm.getelementptr inbounds %31[%369] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %368, %370 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %371 = llvm.add %369, %4 : i64
    llvm.store %371, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %372 = llvm.getelementptr inbounds %arg1[29] : (!llvm.ptr) -> !llvm.ptr, f32
    %373 = llvm.load %372 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %374 = llvm.getelementptr inbounds %arg1[28] : (!llvm.ptr) -> !llvm.ptr, f32
    %375 = llvm.load %374 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %376 = llvm.getelementptr inbounds %arg1[27] : (!llvm.ptr) -> !llvm.ptr, f32
    %377 = llvm.load %376 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %378 = llvm.fpext %377 : f32 to f64
    %379 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %380 = llvm.getelementptr inbounds %31[%379] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %378, %380 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %381 = llvm.add %379, %4 : i64
    llvm.store %381, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %382 = llvm.fpext %375 : f32 to f64
    %383 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %384 = llvm.getelementptr inbounds %31[%383] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %382, %384 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %385 = llvm.add %383, %4 : i64
    llvm.store %385, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %386 = llvm.fpext %373 : f32 to f64
    %387 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %388 = llvm.getelementptr inbounds %31[%387] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %386, %388 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %389 = llvm.add %387, %4 : i64
    llvm.store %389, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %390 = llvm.getelementptr inbounds %arg1[35] : (!llvm.ptr) -> !llvm.ptr, f32
    %391 = llvm.load %390 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %392 = llvm.getelementptr inbounds %arg1[34] : (!llvm.ptr) -> !llvm.ptr, f32
    %393 = llvm.load %392 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %394 = llvm.getelementptr inbounds %arg1[33] : (!llvm.ptr) -> !llvm.ptr, f32
    %395 = llvm.load %394 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %396 = llvm.fpext %395 : f32 to f64
    %397 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %398 = llvm.getelementptr inbounds %31[%397] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %396, %398 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %399 = llvm.add %397, %4 : i64
    llvm.store %399, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %400 = llvm.fpext %393 : f32 to f64
    %401 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %402 = llvm.getelementptr inbounds %31[%401] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %400, %402 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %403 = llvm.add %401, %4 : i64
    llvm.store %403, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %404 = llvm.fpext %391 : f32 to f64
    %405 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %406 = llvm.getelementptr inbounds %31[%405] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %404, %406 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %407 = llvm.add %405, %4 : i64
    llvm.store %407, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %408 = llvm.fpext %353 : f32 to f64
    %409 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %410 = llvm.getelementptr inbounds %31[%409] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %408, %410 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %411 = llvm.add %409, %4 : i64
    llvm.store %411, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %412 = llvm.fpext %351 : f32 to f64
    %413 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %414 = llvm.getelementptr inbounds %31[%413] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %412, %414 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %415 = llvm.add %413, %4 : i64
    llvm.store %415, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %416 = llvm.fpext %349 : f32 to f64
    %417 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %418 = llvm.getelementptr inbounds %31[%417] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %416, %418 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %419 = llvm.add %417, %4 : i64
    llvm.store %419, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %420 = llvm.getelementptr inbounds %arg1[56] : (!llvm.ptr) -> !llvm.ptr, f32
    %421 = llvm.load %420 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %422 = llvm.getelementptr inbounds %arg1[55] : (!llvm.ptr) -> !llvm.ptr, f32
    %423 = llvm.load %422 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %424 = llvm.getelementptr inbounds %arg1[54] : (!llvm.ptr) -> !llvm.ptr, f32
    %425 = llvm.load %424 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %426 = llvm.fpext %425 : f32 to f64
    %427 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %428 = llvm.getelementptr inbounds %31[%427] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %426, %428 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %429 = llvm.add %427, %4 : i64
    llvm.store %429, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %430 = llvm.fpext %423 : f32 to f64
    %431 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %432 = llvm.getelementptr inbounds %31[%431] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %430, %432 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %433 = llvm.add %431, %4 : i64
    llvm.store %433, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %434 = llvm.fpext %421 : f32 to f64
    %435 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %436 = llvm.getelementptr inbounds %31[%435] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %434, %436 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %437 = llvm.add %435, %4 : i64
    llvm.store %437, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %438 = llvm.getelementptr inbounds %arg1[65] : (!llvm.ptr) -> !llvm.ptr, f32
    %439 = llvm.load %438 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %440 = llvm.getelementptr inbounds %arg1[64] : (!llvm.ptr) -> !llvm.ptr, f32
    %441 = llvm.load %440 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %442 = llvm.getelementptr inbounds %arg1[63] : (!llvm.ptr) -> !llvm.ptr, f32
    %443 = llvm.load %442 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %444 = llvm.getelementptr inbounds %arg1[47] : (!llvm.ptr) -> !llvm.ptr, f32
    %445 = llvm.load %444 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %446 = llvm.getelementptr inbounds %arg1[46] : (!llvm.ptr) -> !llvm.ptr, f32
    %447 = llvm.load %446 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %448 = llvm.getelementptr inbounds %arg1[45] : (!llvm.ptr) -> !llvm.ptr, f32
    %449 = llvm.load %448 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %450 = llvm.fpext %449 : f32 to f64
    %451 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %452 = llvm.getelementptr inbounds %31[%451] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %450, %452 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %453 = llvm.add %451, %4 : i64
    llvm.store %453, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %454 = llvm.fpext %447 : f32 to f64
    %455 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %456 = llvm.getelementptr inbounds %31[%455] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %454, %456 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %457 = llvm.add %455, %4 : i64
    llvm.store %457, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %458 = llvm.fpext %445 : f32 to f64
    %459 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %460 = llvm.getelementptr inbounds %31[%459] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %458, %460 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %461 = llvm.add %459, %4 : i64
    llvm.store %461, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %462 = llvm.fpext %443 : f32 to f64
    %463 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %464 = llvm.getelementptr inbounds %31[%463] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %462, %464 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %465 = llvm.add %463, %4 : i64
    llvm.store %465, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %466 = llvm.fpext %441 : f32 to f64
    %467 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %468 = llvm.getelementptr inbounds %31[%467] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %466, %468 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %469 = llvm.add %467, %4 : i64
    llvm.store %469, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %470 = llvm.fpext %439 : f32 to f64
    %471 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %472 = llvm.getelementptr inbounds %31[%471] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %470, %472 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %473 = llvm.add %471, %4 : i64
    llvm.store %473, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %474 = llvm.fpext %43 : f32 to f64
    %475 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %476 = llvm.getelementptr inbounds %31[%475] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %474, %476 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %477 = llvm.add %475, %4 : i64
    llvm.store %477, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %478 = llvm.fpext %41 : f32 to f64
    %479 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %480 = llvm.getelementptr inbounds %31[%479] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %478, %480 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %481 = llvm.add %479, %4 : i64
    llvm.store %481, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %482 = llvm.fpext %39 : f32 to f64
    %483 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %484 = llvm.getelementptr inbounds %31[%483] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %482, %484 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %485 = llvm.add %483, %4 : i64
    llvm.store %485, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %486 = llvm.getelementptr inbounds %arg1[86] : (!llvm.ptr) -> !llvm.ptr, f32
    %487 = llvm.load %486 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %488 = llvm.getelementptr inbounds %arg1[85] : (!llvm.ptr) -> !llvm.ptr, f32
    %489 = llvm.load %488 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %490 = llvm.getelementptr inbounds %arg1[84] : (!llvm.ptr) -> !llvm.ptr, f32
    %491 = llvm.load %490 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %492 = llvm.getelementptr inbounds %arg1[71] : (!llvm.ptr) -> !llvm.ptr, f32
    %493 = llvm.load %492 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %494 = llvm.getelementptr inbounds %arg1[70] : (!llvm.ptr) -> !llvm.ptr, f32
    %495 = llvm.load %494 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %496 = llvm.getelementptr inbounds %arg1[69] : (!llvm.ptr) -> !llvm.ptr, f32
    %497 = llvm.load %496 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %498 = llvm.fpext %497 : f32 to f64
    %499 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %500 = llvm.getelementptr inbounds %31[%499] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %498, %500 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %501 = llvm.add %499, %4 : i64
    llvm.store %501, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %502 = llvm.fpext %495 : f32 to f64
    %503 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %504 = llvm.getelementptr inbounds %31[%503] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %502, %504 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %505 = llvm.add %503, %4 : i64
    llvm.store %505, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %506 = llvm.fpext %493 : f32 to f64
    %507 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %508 = llvm.getelementptr inbounds %31[%507] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %506, %508 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %509 = llvm.add %507, %4 : i64
    llvm.store %509, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %510 = llvm.getelementptr inbounds %arg1[53] : (!llvm.ptr) -> !llvm.ptr, f32
    %511 = llvm.load %510 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %512 = llvm.getelementptr inbounds %arg1[52] : (!llvm.ptr) -> !llvm.ptr, f32
    %513 = llvm.load %512 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %514 = llvm.getelementptr inbounds %arg1[51] : (!llvm.ptr) -> !llvm.ptr, f32
    %515 = llvm.load %514 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %516 = llvm.fpext %515 : f32 to f64
    %517 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %518 = llvm.getelementptr inbounds %31[%517] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %516, %518 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %519 = llvm.add %517, %4 : i64
    llvm.store %519, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %520 = llvm.fpext %513 : f32 to f64
    %521 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %522 = llvm.getelementptr inbounds %31[%521] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %520, %522 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %523 = llvm.add %521, %4 : i64
    llvm.store %523, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %524 = llvm.fpext %511 : f32 to f64
    %525 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %526 = llvm.getelementptr inbounds %31[%525] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %524, %526 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %527 = llvm.add %525, %4 : i64
    llvm.store %527, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %528 = llvm.getelementptr inbounds %arg1[62] : (!llvm.ptr) -> !llvm.ptr, f32
    %529 = llvm.load %528 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %530 = llvm.getelementptr inbounds %arg1[61] : (!llvm.ptr) -> !llvm.ptr, f32
    %531 = llvm.load %530 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %532 = llvm.getelementptr inbounds %arg1[60] : (!llvm.ptr) -> !llvm.ptr, f32
    %533 = llvm.load %532 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %534 = llvm.fpext %533 : f32 to f64
    %535 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %536 = llvm.getelementptr inbounds %31[%535] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %534, %536 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %537 = llvm.add %535, %4 : i64
    llvm.store %537, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %538 = llvm.fpext %531 : f32 to f64
    %539 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %540 = llvm.getelementptr inbounds %31[%539] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %538, %540 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %541 = llvm.add %539, %4 : i64
    llvm.store %541, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %542 = llvm.fpext %529 : f32 to f64
    %543 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %544 = llvm.getelementptr inbounds %31[%543] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %542, %544 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %545 = llvm.add %543, %4 : i64
    llvm.store %545, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %546 = llvm.fpext %491 : f32 to f64
    %547 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %548 = llvm.getelementptr inbounds %31[%547] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %546, %548 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %549 = llvm.add %547, %4 : i64
    llvm.store %549, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %550 = llvm.fpext %489 : f32 to f64
    %551 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %552 = llvm.getelementptr inbounds %31[%551] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %550, %552 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %553 = llvm.add %551, %4 : i64
    llvm.store %553, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %554 = llvm.fpext %487 : f32 to f64
    %555 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %556 = llvm.getelementptr inbounds %31[%555] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %554, %556 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %557 = llvm.add %555, %4 : i64
    llvm.store %557, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %558 = llvm.getelementptr inbounds %arg1[77] : (!llvm.ptr) -> !llvm.ptr, f32
    %559 = llvm.load %558 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %560 = llvm.getelementptr inbounds %arg1[76] : (!llvm.ptr) -> !llvm.ptr, f32
    %561 = llvm.load %560 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %562 = llvm.getelementptr inbounds %arg1[75] : (!llvm.ptr) -> !llvm.ptr, f32
    %563 = llvm.load %562 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %564 = llvm.getelementptr inbounds %arg1[68] : (!llvm.ptr) -> !llvm.ptr, f32
    %565 = llvm.load %564 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %566 = llvm.getelementptr inbounds %arg1[67] : (!llvm.ptr) -> !llvm.ptr, f32
    %567 = llvm.load %566 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %568 = llvm.getelementptr inbounds %arg1[66] : (!llvm.ptr) -> !llvm.ptr, f32
    %569 = llvm.load %568 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %570 = llvm.fpext %569 : f32 to f64
    %571 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %572 = llvm.getelementptr inbounds %31[%571] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %570, %572 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %573 = llvm.add %571, %4 : i64
    llvm.store %573, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %574 = llvm.fpext %567 : f32 to f64
    %575 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %576 = llvm.getelementptr inbounds %31[%575] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %574, %576 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %577 = llvm.add %575, %4 : i64
    llvm.store %577, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %578 = llvm.fpext %565 : f32 to f64
    %579 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %580 = llvm.getelementptr inbounds %31[%579] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %578, %580 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %581 = llvm.add %579, %4 : i64
    llvm.store %581, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %582 = llvm.fpext %563 : f32 to f64
    %583 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %584 = llvm.getelementptr inbounds %31[%583] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %582, %584 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %585 = llvm.add %583, %4 : i64
    llvm.store %585, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %586 = llvm.fpext %561 : f32 to f64
    %587 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %588 = llvm.getelementptr inbounds %31[%587] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %586, %588 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %589 = llvm.add %587, %4 : i64
    llvm.store %589, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %590 = llvm.fpext %559 : f32 to f64
    %591 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %592 = llvm.getelementptr inbounds %31[%591] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %590, %592 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %593 = llvm.add %591, %4 : i64
    llvm.store %593, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %594 = llvm.getelementptr inbounds %arg1[89] : (!llvm.ptr) -> !llvm.ptr, f32
    %595 = llvm.load %594 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %596 = llvm.getelementptr inbounds %arg1[88] : (!llvm.ptr) -> !llvm.ptr, f32
    %597 = llvm.load %596 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %598 = llvm.getelementptr inbounds %arg1[87] : (!llvm.ptr) -> !llvm.ptr, f32
    %599 = llvm.load %598 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %600 = llvm.fpext %599 : f32 to f64
    %601 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %602 = llvm.getelementptr inbounds %31[%601] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %600, %602 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %603 = llvm.add %601, %4 : i64
    llvm.store %603, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %604 = llvm.fpext %597 : f32 to f64
    %605 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %606 = llvm.getelementptr inbounds %31[%605] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %604, %606 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %607 = llvm.add %605, %4 : i64
    llvm.store %607, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %608 = llvm.fpext %595 : f32 to f64
    %609 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %610 = llvm.getelementptr inbounds %31[%609] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %608, %610 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %611 = llvm.add %609, %4 : i64
    llvm.store %611, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %612 = llvm.getelementptr inbounds %arg1[80] : (!llvm.ptr) -> !llvm.ptr, f32
    %613 = llvm.load %612 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %614 = llvm.getelementptr inbounds %arg1[79] : (!llvm.ptr) -> !llvm.ptr, f32
    %615 = llvm.load %614 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %616 = llvm.getelementptr inbounds %arg1[78] : (!llvm.ptr) -> !llvm.ptr, f32
    %617 = llvm.load %616 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %618 = llvm.fpext %617 : f32 to f64
    %619 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %620 = llvm.getelementptr inbounds %31[%619] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %618, %620 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %621 = llvm.add %619, %4 : i64
    llvm.store %621, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %622 = llvm.fpext %615 : f32 to f64
    %623 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %624 = llvm.getelementptr inbounds %31[%623] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %622, %624 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %625 = llvm.add %623, %4 : i64
    llvm.store %625, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %626 = llvm.fpext %613 : f32 to f64
    %627 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %628 = llvm.getelementptr inbounds %31[%627] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %626, %628 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %629 = llvm.add %627, %4 : i64
    llvm.store %629, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %630 = llvm.getelementptr inbounds %arg1[92] : (!llvm.ptr) -> !llvm.ptr, f32
    %631 = llvm.load %630 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %632 = llvm.getelementptr inbounds %arg1[91] : (!llvm.ptr) -> !llvm.ptr, f32
    %633 = llvm.load %632 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %634 = llvm.getelementptr inbounds %arg1[90] : (!llvm.ptr) -> !llvm.ptr, f32
    %635 = llvm.load %634 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %636 = llvm.fpext %635 : f32 to f64
    %637 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %638 = llvm.getelementptr inbounds %31[%637] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %636, %638 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %639 = llvm.add %637, %4 : i64
    llvm.store %639, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %640 = llvm.fpext %633 : f32 to f64
    %641 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %642 = llvm.getelementptr inbounds %31[%641] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %640, %642 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %643 = llvm.add %641, %4 : i64
    llvm.store %643, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %644 = llvm.fpext %631 : f32 to f64
    %645 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %646 = llvm.getelementptr inbounds %31[%645] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %644, %646 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %647 = llvm.add %645, %4 : i64
    llvm.store %647, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %648 = llvm.getelementptr inbounds %arg1[83] : (!llvm.ptr) -> !llvm.ptr, f32
    %649 = llvm.load %648 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %650 = llvm.getelementptr inbounds %arg1[82] : (!llvm.ptr) -> !llvm.ptr, f32
    %651 = llvm.load %650 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %652 = llvm.getelementptr inbounds %arg1[81] : (!llvm.ptr) -> !llvm.ptr, f32
    %653 = llvm.load %652 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %654 = llvm.fpext %653 : f32 to f64
    %655 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %656 = llvm.getelementptr inbounds %31[%655] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %654, %656 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %657 = llvm.add %655, %4 : i64
    llvm.store %657, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %658 = llvm.fpext %651 : f32 to f64
    %659 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %660 = llvm.getelementptr inbounds %31[%659] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %658, %660 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %661 = llvm.add %659, %4 : i64
    llvm.store %661, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %662 = llvm.fpext %649 : f32 to f64
    %663 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %664 = llvm.getelementptr inbounds %31[%663] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %662, %664 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %665 = llvm.add %663, %4 : i64
    llvm.store %665, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %666 = llvm.getelementptr inbounds %arg1[95] : (!llvm.ptr) -> !llvm.ptr, f32
    %667 = llvm.load %666 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %668 = llvm.getelementptr inbounds %arg1[94] : (!llvm.ptr) -> !llvm.ptr, f32
    %669 = llvm.load %668 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %670 = llvm.getelementptr inbounds %arg1[93] : (!llvm.ptr) -> !llvm.ptr, f32
    %671 = llvm.load %670 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %672 = llvm.fpext %671 : f32 to f64
    %673 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %674 = llvm.getelementptr inbounds %31[%673] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %672, %674 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %675 = llvm.add %673, %4 : i64
    llvm.store %675, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %676 = llvm.fpext %669 : f32 to f64
    %677 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %678 = llvm.getelementptr inbounds %31[%677] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %676, %678 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %679 = llvm.add %677, %4 : i64
    llvm.store %679, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %680 = llvm.fpext %667 : f32 to f64
    %681 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %682 = llvm.getelementptr inbounds %31[%681] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %680, %682 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %683 = llvm.add %681, %4 : i64
    llvm.store %683, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    llvm.store %24, %28 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>, !llvm.ptr
    llvm.store %15, %27 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    llvm.store %36, %26 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %684 = llvm.getelementptr %2[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %685 = llvm.ptrtoint %684 : !llvm.ptr to i64
    %686 = llvm.call @_mlir_memref_to_llvm_alloc(%685) : (i64) -> !llvm.ptr
    %687 = llvm.insertvalue %686, %10[0] : !llvm.struct<(ptr, ptr, i64)> 
    %688 = llvm.insertvalue %686, %687[1] : !llvm.struct<(ptr, ptr, i64)> 
    %689 = llvm.insertvalue %5, %688[2] : !llvm.struct<(ptr, ptr, i64)> 
    llvm.store %689, %25 : !llvm.struct<(ptr, ptr, i64)>, !llvm.ptr
    llvm.call @qnode_forward_0.quantum(%28, %27, %26, %25) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %690 = llvm.load %686 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.store %690, %arg16 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    llvm.return
  }
  llvm.func @setup() {
    %0 = llvm.mlir.zero : !llvm.ptr
    llvm.call @__catalyst__rt__initialize(%0) : (!llvm.ptr) -> ()
    llvm.return
  }
  llvm.func @teardown() {
    llvm.call @__catalyst__rt__finalize() : () -> ()
    llvm.return
  }
}