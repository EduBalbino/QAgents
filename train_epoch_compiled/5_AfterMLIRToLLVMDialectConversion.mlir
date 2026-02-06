module @train_epoch_compiled {
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
  llvm.mlir.global internal constant @"{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}"("{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @LightningSimulator("LightningSimulator\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @"/home/pichau/QAgents/.venv/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.so"("/home/pichau/QAgents/.venv/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.so\00") {addr_space = 0 : i32}
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
  llvm.mlir.global private constant @__constant_xi64_4(32 : i64) {addr_space = 0 : i32, alignment = 64 : i64} : i64
  llvm.mlir.global private constant @__constant_4xi32_3(dense<[13, 15, 26, 6]> : tensor<4xi32>) {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<4 x i32>
  llvm.mlir.global private constant @__constant_4xi32(dense<[17, 29, 16, 24]> : tensor<4xi32>) {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<4 x i32>
  llvm.mlir.global private constant @__constant_xi64(1 : i64) {addr_space = 0 : i32, alignment = 64 : i64} : i64
  llvm.mlir.global private constant @__constant_xf32_2(0.000000e+00 : f32) {addr_space = 0 : i32, alignment = 64 : i64} : f32
  llvm.mlir.global private constant @__constant_xf32_1(0.00999999977 : f32) {addr_space = 0 : i32, alignment = 64 : i64} : f32
  llvm.mlir.global private constant @__constant_xf32_0(3.125000e-02 : f32) {addr_space = 0 : i32, alignment = 64 : i64} : f32
  llvm.mlir.global private constant @__constant_xf32(3.14159274 : f32) {addr_space = 0 : i32, alignment = 64 : i64} : f32
  llvm.func @jit_train_epoch_compiled(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: !llvm.ptr, %arg10: !llvm.ptr, %arg11: i64, %arg12: !llvm.ptr, %arg13: !llvm.ptr, %arg14: i64, %arg15: !llvm.ptr, %arg16: !llvm.ptr, %arg17: i64, %arg18: !llvm.ptr, %arg19: !llvm.ptr, %arg20: i64, %arg21: i64, %arg22: i64, %arg23: !llvm.ptr, %arg24: !llvm.ptr, %arg25: i64, %arg26: i64, %arg27: i64, %arg28: i64, %arg29: i64, %arg30: !llvm.ptr, %arg31: !llvm.ptr, %arg32: i64, %arg33: i64, %arg34: i64, %arg35: !llvm.ptr, %arg36: !llvm.ptr, %arg37: i64, %arg38: i64, %arg39: i64) -> !llvm.struct<(struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>, struct<(ptr, ptr, i64)>, struct<(ptr, ptr, i64)>, struct<(ptr, ptr, i64)>, struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)> attributes {llvm.copy_memref, llvm.emit_c_interface} {
    %0 = llvm.mlir.poison : !llvm.struct<(struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>, struct<(ptr, ptr, i64)>, struct<(ptr, ptr, i64)>, struct<(ptr, ptr, i64)>, struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>
    %1 = llvm.mlir.constant(false) : i1
    %2 = llvm.mlir.constant(24 : index) : i64
    %3 = llvm.mlir.constant(64 : index) : i64
    %4 = llvm.mlir.addressof @__constant_xi64_4 : !llvm.ptr
    %5 = llvm.mlir.addressof @__constant_4xi32_3 : !llvm.ptr
    %6 = llvm.mlir.addressof @__constant_4xi32 : !llvm.ptr
    %7 = llvm.mlir.addressof @__constant_xi64 : !llvm.ptr
    %8 = llvm.mlir.addressof @__constant_xf32_2 : !llvm.ptr
    %9 = llvm.mlir.addressof @__constant_xf32_1 : !llvm.ptr
    %10 = llvm.mlir.addressof @__constant_xf32_0 : !llvm.ptr
    %11 = llvm.mlir.zero : !llvm.ptr
    %12 = llvm.mlir.constant(0.000000e+00 : f64) : f64
    %13 = llvm.mlir.constant(1.000000e+00 : f64) : f64
    %14 = llvm.mlir.constant(0 : index) : i64
    %15 = llvm.mlir.constant(3 : index) : i64
    %16 = llvm.mlir.constant(32 : index) : i64
    %17 = llvm.mlir.constant(32 : i64) : i64
    %18 = llvm.mlir.constant(0 : i64) : i64
    %19 = llvm.mlir.constant(96 : i64) : i64
    %20 = llvm.mlir.constant(95 : index) : i64
    %21 = llvm.mlir.constant(0.00999999977 : f32) : f32
    %22 = llvm.mlir.constant(3.200000e+01 : f64) : f64
    %23 = llvm.mlir.constant(3.000000e+00 : f64) : f64
    %24 = llvm.mlir.constant(3.125000e-02 : f64) : f64
    %25 = llvm.mlir.constant(0 : i32) : i32
    %26 = llvm.mlir.constant(32 : i32) : i32
    %27 = llvm.mlir.constant(5 : i64) : i64
    %28 = llvm.mlir.constant(1 : i64) : i64
    %29 = llvm.mlir.constant(64 : i64) : i64
    %30 = llvm.mlir.constant(466688986 : i32) : i32
    %31 = llvm.mlir.constant(3735928559 : index) : i64
    %32 = llvm.mlir.addressof @_sample_loss.cloned : !llvm.ptr
    %33 = llvm.mlir.addressof @enzyme_const : !llvm.ptr
    %34 = llvm.mlir.addressof @enzyme_dupnoneed : !llvm.ptr
    %35 = llvm.mlir.constant(0 : i8) : i8
    %36 = llvm.mlir.constant(1 : index) : i64
    %37 = llvm.mlir.constant(2 : index) : i64
    %38 = llvm.mlir.constant(4 : index) : i64
    %39 = llvm.mlir.constant(8 : index) : i64
    %40 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %41 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64)>
    %42 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %43 = llvm.insertvalue %arg15, %41[0] : !llvm.struct<(ptr, ptr, i64)> 
    %44 = llvm.insertvalue %arg16, %43[1] : !llvm.struct<(ptr, ptr, i64)> 
    %45 = llvm.insertvalue %arg17, %44[2] : !llvm.struct<(ptr, ptr, i64)> 
    %46 = llvm.getelementptr inbounds %6[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i32>
    %47 = llvm.getelementptr inbounds %5[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i32>
    %48 = llvm.load %arg19 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %49 = llvm.getelementptr inbounds %arg19[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %50 = llvm.load %49 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %51 = llvm.getelementptr %11[2] : (!llvm.ptr) -> !llvm.ptr, i64
    %52 = llvm.ptrtoint %51 : !llvm.ptr to i64
    %53 = llvm.add %52, %3 : i64
    %54 = llvm.call @_mlir_memref_to_llvm_alloc(%53) : (i64) -> !llvm.ptr
    %55 = llvm.ptrtoint %54 : !llvm.ptr to i64
    %56 = llvm.sub %3, %36 : i64
    %57 = llvm.add %55, %56 : i64
    %58 = llvm.urem %57, %3 : i64
    %59 = llvm.sub %57, %58 : i64
    %60 = llvm.inttoptr %59 : i64 to !llvm.ptr
    %61 = llvm.getelementptr %11[2] : (!llvm.ptr) -> !llvm.ptr, i64
    %62 = llvm.ptrtoint %61 : !llvm.ptr to i64
    %63 = llvm.add %62, %3 : i64
    %64 = llvm.call @_mlir_memref_to_llvm_alloc(%63) : (i64) -> !llvm.ptr
    %65 = llvm.ptrtoint %64 : !llvm.ptr to i64
    %66 = llvm.sub %3, %36 : i64
    %67 = llvm.add %65, %66 : i64
    %68 = llvm.urem %67, %3 : i64
    %69 = llvm.sub %67, %68 : i64
    %70 = llvm.inttoptr %69 : i64 to !llvm.ptr
    llvm.br ^bb1(%14 : i64)
  ^bb1(%71: i64):  // 2 preds: ^bb0, ^bb2
    %72 = llvm.icmp "slt" %71, %37 : i64
    llvm.cond_br %72, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %73 = llvm.getelementptr inbounds %70[%71] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %71, %73 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %74 = llvm.add %71, %36 : i64
    llvm.br ^bb1(%74 : i64)
  ^bb3:  // pred: ^bb1
    llvm.br ^bb4(%14 : i64)
  ^bb4(%75: i64):  // 2 preds: ^bb3, ^bb5
    %76 = llvm.icmp "slt" %75, %37 : i64
    llvm.cond_br %76, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %77 = llvm.load %7 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %78 = llvm.getelementptr inbounds %60[%75] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %77, %78 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %79 = llvm.add %75, %36 : i64
    llvm.br ^bb4(%79 : i64)
  ^bb6:  // pred: ^bb4
    %80 = llvm.getelementptr %11[2] : (!llvm.ptr) -> !llvm.ptr, i64
    %81 = llvm.ptrtoint %80 : !llvm.ptr to i64
    %82 = llvm.add %81, %3 : i64
    %83 = llvm.call @_mlir_memref_to_llvm_alloc(%82) : (i64) -> !llvm.ptr
    %84 = llvm.ptrtoint %83 : !llvm.ptr to i64
    %85 = llvm.sub %3, %36 : i64
    %86 = llvm.add %84, %85 : i64
    %87 = llvm.urem %86, %3 : i64
    %88 = llvm.sub %86, %87 : i64
    %89 = llvm.inttoptr %88 : i64 to !llvm.ptr
    llvm.br ^bb7(%14 : i64)
  ^bb7(%90: i64):  // 2 preds: ^bb6, ^bb8
    %91 = llvm.icmp "slt" %90, %37 : i64
    llvm.cond_br %91, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    %92 = llvm.getelementptr inbounds %60[%90] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %93 = llvm.load %92 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %94 = llvm.getelementptr inbounds %70[%90] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %95 = llvm.load %94 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %96 = llvm.mul %93, %95 : i64
    %97 = llvm.getelementptr inbounds %89[%90] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %96, %97 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %98 = llvm.add %90, %36 : i64
    llvm.br ^bb7(%98 : i64)
  ^bb9:  // pred: ^bb7
    llvm.call @_mlir_memref_to_llvm_free(%64) : (!llvm.ptr) -> ()
    llvm.br ^bb10(%14 : i64)
  ^bb10(%99: i64):  // 2 preds: ^bb9, ^bb11
    %100 = llvm.icmp "slt" %99, %37 : i64
    llvm.cond_br %100, ^bb11, ^bb12
  ^bb11:  // pred: ^bb10
    %101 = llvm.load %4 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %102 = llvm.getelementptr inbounds %60[%99] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %101, %102 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %103 = llvm.add %99, %36 : i64
    llvm.br ^bb10(%103 : i64)
  ^bb12:  // pred: ^bb10
    llvm.br ^bb13(%14 : i64)
  ^bb13(%104: i64):  // 2 preds: ^bb12, ^bb14
    %105 = llvm.icmp "slt" %104, %37 : i64
    llvm.cond_br %105, ^bb14, ^bb15
  ^bb14:  // pred: ^bb13
    %106 = llvm.getelementptr inbounds %89[%104] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %107 = llvm.load %106 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %108 = llvm.getelementptr inbounds %60[%104] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %109 = llvm.load %108 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %110 = llvm.lshr %107, %109 : i64
    %111 = llvm.icmp "ult" %109, %29 : i64
    %112 = llvm.select %111, %110, %18 : i1, i64
    %113 = llvm.getelementptr inbounds %60[%104] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %112, %113 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %114 = llvm.add %104, %36 : i64
    llvm.br ^bb13(%114 : i64)
  ^bb15:  // pred: ^bb13
    %115 = llvm.getelementptr %11[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %116 = llvm.ptrtoint %115 : !llvm.ptr to i64
    %117 = llvm.add %116, %3 : i64
    %118 = llvm.call @_mlir_memref_to_llvm_alloc(%117) : (i64) -> !llvm.ptr
    %119 = llvm.ptrtoint %118 : !llvm.ptr to i64
    %120 = llvm.sub %3, %36 : i64
    %121 = llvm.add %119, %120 : i64
    %122 = llvm.urem %121, %3 : i64
    %123 = llvm.sub %121, %122 : i64
    %124 = llvm.inttoptr %123 : i64 to !llvm.ptr
    %125 = llvm.getelementptr %11[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %126 = llvm.ptrtoint %125 : !llvm.ptr to i64
    %127 = llvm.add %126, %3 : i64
    %128 = llvm.call @_mlir_memref_to_llvm_alloc(%127) : (i64) -> !llvm.ptr
    %129 = llvm.ptrtoint %128 : !llvm.ptr to i64
    %130 = llvm.sub %3, %36 : i64
    %131 = llvm.add %129, %130 : i64
    %132 = llvm.urem %131, %3 : i64
    %133 = llvm.sub %131, %132 : i64
    %134 = llvm.inttoptr %133 : i64 to !llvm.ptr
    llvm.br ^bb16(%14 : i64)
  ^bb16(%135: i64):  // 2 preds: ^bb15, ^bb17
    %136 = llvm.icmp "slt" %135, %37 : i64
    llvm.cond_br %136, ^bb17, ^bb18
  ^bb17:  // pred: ^bb16
    %137 = llvm.getelementptr inbounds %89[%135] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %138 = llvm.load %137 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %139 = llvm.trunc %138 : i64 to i32
    %140 = llvm.getelementptr inbounds %134[%135] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %139, %140 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %141 = llvm.add %135, %36 : i64
    llvm.br ^bb16(%141 : i64)
  ^bb18:  // pred: ^bb16
    llvm.call @_mlir_memref_to_llvm_free(%83) : (!llvm.ptr) -> ()
    %142 = llvm.getelementptr %11[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %143 = llvm.ptrtoint %142 : !llvm.ptr to i64
    %144 = llvm.add %143, %3 : i64
    %145 = llvm.call @_mlir_memref_to_llvm_alloc(%144) : (i64) -> !llvm.ptr
    %146 = llvm.ptrtoint %145 : !llvm.ptr to i64
    %147 = llvm.sub %3, %36 : i64
    %148 = llvm.add %146, %147 : i64
    %149 = llvm.urem %148, %3 : i64
    %150 = llvm.sub %148, %149 : i64
    %151 = llvm.inttoptr %150 : i64 to !llvm.ptr
    llvm.br ^bb19(%14 : i64)
  ^bb19(%152: i64):  // 2 preds: ^bb18, ^bb20
    %153 = llvm.icmp "slt" %152, %37 : i64
    llvm.cond_br %153, ^bb20, ^bb21
  ^bb20:  // pred: ^bb19
    %154 = llvm.getelementptr inbounds %60[%152] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %155 = llvm.load %154 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %156 = llvm.trunc %155 : i64 to i32
    %157 = llvm.getelementptr inbounds %151[%152] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %156, %157 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %158 = llvm.add %152, %36 : i64
    llvm.br ^bb19(%158 : i64)
  ^bb21:  // pred: ^bb19
    llvm.call @_mlir_memref_to_llvm_free(%54) : (!llvm.ptr) -> ()
    %159 = llvm.xor %48, %50 : i32
    %160 = llvm.xor %159, %30 : i32
    llvm.br ^bb22(%14 : i64)
  ^bb22(%161: i64):  // 2 preds: ^bb21, ^bb23
    %162 = llvm.icmp "slt" %161, %37 : i64
    llvm.cond_br %162, ^bb23, ^bb24
  ^bb23:  // pred: ^bb22
    %163 = llvm.load %arg19 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %164 = llvm.getelementptr inbounds %124[%161] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %163, %164 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %165 = llvm.add %161, %36 : i64
    llvm.br ^bb22(%165 : i64)
  ^bb24:  // pred: ^bb22
    %166 = llvm.getelementptr %11[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %167 = llvm.ptrtoint %166 : !llvm.ptr to i64
    %168 = llvm.add %167, %3 : i64
    %169 = llvm.call @_mlir_memref_to_llvm_alloc(%168) : (i64) -> !llvm.ptr
    %170 = llvm.ptrtoint %169 : !llvm.ptr to i64
    %171 = llvm.sub %3, %36 : i64
    %172 = llvm.add %170, %171 : i64
    %173 = llvm.urem %172, %3 : i64
    %174 = llvm.sub %172, %173 : i64
    %175 = llvm.inttoptr %174 : i64 to !llvm.ptr
    llvm.br ^bb25(%14 : i64)
  ^bb25(%176: i64):  // 2 preds: ^bb24, ^bb26
    %177 = llvm.icmp "slt" %176, %37 : i64
    llvm.cond_br %177, ^bb26, ^bb27
  ^bb26:  // pred: ^bb25
    %178 = llvm.getelementptr inbounds %151[%176] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %179 = llvm.load %178 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %180 = llvm.getelementptr inbounds %124[%176] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %181 = llvm.load %180 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %182 = llvm.add %179, %181 : i32
    %183 = llvm.getelementptr inbounds %175[%176] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %182, %183 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %184 = llvm.add %176, %36 : i64
    llvm.br ^bb25(%184 : i64)
  ^bb27:  // pred: ^bb25
    llvm.call @_mlir_memref_to_llvm_free(%145) : (!llvm.ptr) -> ()
    llvm.br ^bb28(%14 : i64)
  ^bb28(%185: i64):  // 2 preds: ^bb27, ^bb29
    %186 = llvm.icmp "slt" %185, %37 : i64
    llvm.cond_br %186, ^bb29, ^bb30
  ^bb29:  // pred: ^bb28
    %187 = llvm.getelementptr inbounds %arg19[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %188 = llvm.load %187 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %189 = llvm.getelementptr inbounds %124[%185] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %188, %189 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %190 = llvm.add %185, %36 : i64
    llvm.br ^bb28(%190 : i64)
  ^bb30:  // pred: ^bb28
    llvm.br ^bb31(%14 : i64)
  ^bb31(%191: i64):  // 2 preds: ^bb30, ^bb32
    %192 = llvm.icmp "slt" %191, %37 : i64
    llvm.cond_br %192, ^bb32, ^bb33
  ^bb32:  // pred: ^bb31
    %193 = llvm.getelementptr inbounds %134[%191] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %194 = llvm.load %193 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %195 = llvm.getelementptr inbounds %124[%191] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %196 = llvm.load %195 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %197 = llvm.add %194, %196 : i32
    %198 = llvm.getelementptr inbounds %124[%191] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %197, %198 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %199 = llvm.add %191, %36 : i64
    llvm.br ^bb31(%199 : i64)
  ^bb33:  // pred: ^bb31
    llvm.call @_mlir_memref_to_llvm_free(%128) : (!llvm.ptr) -> ()
    %200 = llvm.getelementptr inbounds %arg19[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %201 = llvm.load %200 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %202 = llvm.load %arg19 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %203 = llvm.getelementptr %11[4] : (!llvm.ptr) -> !llvm.ptr, i32
    %204 = llvm.ptrtoint %203 : !llvm.ptr to i64
    %205 = llvm.add %204, %3 : i64
    %206 = llvm.call @_mlir_memref_to_llvm_alloc(%205) : (i64) -> !llvm.ptr
    %207 = llvm.ptrtoint %206 : !llvm.ptr to i64
    %208 = llvm.sub %3, %36 : i64
    %209 = llvm.add %207, %208 : i64
    %210 = llvm.urem %209, %3 : i64
    %211 = llvm.sub %209, %210 : i64
    %212 = llvm.inttoptr %211 : i64 to !llvm.ptr
    %213 = llvm.mul %38, %36 : i64
    %214 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %215 = llvm.ptrtoint %214 : !llvm.ptr to i64
    %216 = llvm.mul %213, %215 : i64
    "llvm.intr.memcpy"(%212, %47, %216) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %217 = llvm.getelementptr %11[4] : (!llvm.ptr) -> !llvm.ptr, i32
    %218 = llvm.ptrtoint %217 : !llvm.ptr to i64
    %219 = llvm.add %218, %3 : i64
    %220 = llvm.call @_mlir_memref_to_llvm_alloc(%219) : (i64) -> !llvm.ptr
    %221 = llvm.ptrtoint %220 : !llvm.ptr to i64
    %222 = llvm.sub %3, %36 : i64
    %223 = llvm.add %221, %222 : i64
    %224 = llvm.urem %223, %3 : i64
    %225 = llvm.sub %223, %224 : i64
    %226 = llvm.inttoptr %225 : i64 to !llvm.ptr
    %227 = llvm.mul %38, %36 : i64
    %228 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %229 = llvm.ptrtoint %228 : !llvm.ptr to i64
    %230 = llvm.mul %227, %229 : i64
    "llvm.intr.memcpy"(%226, %46, %230) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %231 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %232 = llvm.ptrtoint %231 : !llvm.ptr to i64
    %233 = llvm.add %232, %3 : i64
    %234 = llvm.call @_mlir_memref_to_llvm_alloc(%233) : (i64) -> !llvm.ptr
    %235 = llvm.ptrtoint %234 : !llvm.ptr to i64
    %236 = llvm.sub %3, %36 : i64
    %237 = llvm.add %235, %236 : i64
    %238 = llvm.urem %237, %3 : i64
    %239 = llvm.sub %237, %238 : i64
    %240 = llvm.inttoptr %239 : i64 to !llvm.ptr
    %241 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %242 = llvm.ptrtoint %241 : !llvm.ptr to i64
    %243 = llvm.add %242, %3 : i64
    %244 = llvm.call @_mlir_memref_to_llvm_alloc(%243) : (i64) -> !llvm.ptr
    %245 = llvm.ptrtoint %244 : !llvm.ptr to i64
    %246 = llvm.sub %3, %36 : i64
    %247 = llvm.add %245, %246 : i64
    %248 = llvm.urem %247, %3 : i64
    %249 = llvm.sub %247, %248 : i64
    %250 = llvm.inttoptr %249 : i64 to !llvm.ptr
    %251 = llvm.getelementptr %11[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %252 = llvm.ptrtoint %251 : !llvm.ptr to i64
    %253 = llvm.add %252, %3 : i64
    %254 = llvm.call @_mlir_memref_to_llvm_alloc(%253) : (i64) -> !llvm.ptr
    %255 = llvm.ptrtoint %254 : !llvm.ptr to i64
    %256 = llvm.sub %3, %36 : i64
    %257 = llvm.add %255, %256 : i64
    %258 = llvm.urem %257, %3 : i64
    %259 = llvm.sub %257, %258 : i64
    %260 = llvm.inttoptr %259 : i64 to !llvm.ptr
    %261 = llvm.getelementptr %11[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %262 = llvm.ptrtoint %261 : !llvm.ptr to i64
    %263 = llvm.add %262, %3 : i64
    %264 = llvm.call @_mlir_memref_to_llvm_alloc(%263) : (i64) -> !llvm.ptr
    %265 = llvm.ptrtoint %264 : !llvm.ptr to i64
    %266 = llvm.sub %3, %36 : i64
    %267 = llvm.add %265, %266 : i64
    %268 = llvm.urem %267, %3 : i64
    %269 = llvm.sub %267, %268 : i64
    %270 = llvm.inttoptr %269 : i64 to !llvm.ptr
    %271 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %272 = llvm.ptrtoint %271 : !llvm.ptr to i64
    %273 = llvm.add %272, %3 : i64
    %274 = llvm.call @_mlir_memref_to_llvm_alloc(%273) : (i64) -> !llvm.ptr
    %275 = llvm.ptrtoint %274 : !llvm.ptr to i64
    %276 = llvm.sub %3, %36 : i64
    %277 = llvm.add %275, %276 : i64
    %278 = llvm.urem %277, %3 : i64
    %279 = llvm.sub %277, %278 : i64
    %280 = llvm.inttoptr %279 : i64 to !llvm.ptr
    %281 = llvm.getelementptr %11[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %282 = llvm.ptrtoint %281 : !llvm.ptr to i64
    %283 = llvm.add %282, %3 : i64
    %284 = llvm.call @_mlir_memref_to_llvm_alloc(%283) : (i64) -> !llvm.ptr
    %285 = llvm.ptrtoint %284 : !llvm.ptr to i64
    %286 = llvm.sub %3, %36 : i64
    %287 = llvm.add %285, %286 : i64
    %288 = llvm.urem %287, %3 : i64
    %289 = llvm.sub %287, %288 : i64
    %290 = llvm.inttoptr %289 : i64 to !llvm.ptr
    %291 = llvm.getelementptr %11[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %292 = llvm.ptrtoint %291 : !llvm.ptr to i64
    %293 = llvm.add %292, %3 : i64
    %294 = llvm.call @_mlir_memref_to_llvm_alloc(%293) : (i64) -> !llvm.ptr
    %295 = llvm.ptrtoint %294 : !llvm.ptr to i64
    %296 = llvm.sub %3, %36 : i64
    %297 = llvm.add %295, %296 : i64
    %298 = llvm.urem %297, %3 : i64
    %299 = llvm.sub %297, %298 : i64
    %300 = llvm.inttoptr %299 : i64 to !llvm.ptr
    %301 = llvm.getelementptr %11[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %302 = llvm.ptrtoint %301 : !llvm.ptr to i64
    %303 = llvm.add %302, %3 : i64
    %304 = llvm.call @_mlir_memref_to_llvm_alloc(%303) : (i64) -> !llvm.ptr
    %305 = llvm.ptrtoint %304 : !llvm.ptr to i64
    %306 = llvm.sub %3, %36 : i64
    %307 = llvm.add %305, %306 : i64
    %308 = llvm.urem %307, %3 : i64
    %309 = llvm.sub %307, %308 : i64
    %310 = llvm.inttoptr %309 : i64 to !llvm.ptr
    %311 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %312 = llvm.ptrtoint %311 : !llvm.ptr to i64
    %313 = llvm.add %312, %3 : i64
    %314 = llvm.call @_mlir_memref_to_llvm_alloc(%313) : (i64) -> !llvm.ptr
    %315 = llvm.ptrtoint %314 : !llvm.ptr to i64
    %316 = llvm.sub %3, %36 : i64
    %317 = llvm.add %315, %316 : i64
    %318 = llvm.urem %317, %3 : i64
    %319 = llvm.sub %317, %318 : i64
    %320 = llvm.inttoptr %319 : i64 to !llvm.ptr
    %321 = llvm.getelementptr %11[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %322 = llvm.ptrtoint %321 : !llvm.ptr to i64
    %323 = llvm.add %322, %3 : i64
    %324 = llvm.call @_mlir_memref_to_llvm_alloc(%323) : (i64) -> !llvm.ptr
    %325 = llvm.ptrtoint %324 : !llvm.ptr to i64
    %326 = llvm.sub %3, %36 : i64
    %327 = llvm.add %325, %326 : i64
    %328 = llvm.urem %327, %3 : i64
    %329 = llvm.sub %327, %328 : i64
    %330 = llvm.inttoptr %329 : i64 to !llvm.ptr
    %331 = llvm.getelementptr %11[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %332 = llvm.ptrtoint %331 : !llvm.ptr to i64
    %333 = llvm.add %332, %3 : i64
    %334 = llvm.call @_mlir_memref_to_llvm_alloc(%333) : (i64) -> !llvm.ptr
    %335 = llvm.ptrtoint %334 : !llvm.ptr to i64
    %336 = llvm.sub %3, %36 : i64
    %337 = llvm.add %335, %336 : i64
    %338 = llvm.urem %337, %3 : i64
    %339 = llvm.sub %337, %338 : i64
    %340 = llvm.inttoptr %339 : i64 to !llvm.ptr
    %341 = llvm.getelementptr %11[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %342 = llvm.ptrtoint %341 : !llvm.ptr to i64
    %343 = llvm.add %342, %3 : i64
    %344 = llvm.call @_mlir_memref_to_llvm_alloc(%343) : (i64) -> !llvm.ptr
    %345 = llvm.ptrtoint %344 : !llvm.ptr to i64
    %346 = llvm.sub %3, %36 : i64
    %347 = llvm.add %345, %346 : i64
    %348 = llvm.urem %347, %3 : i64
    %349 = llvm.sub %347, %348 : i64
    %350 = llvm.inttoptr %349 : i64 to !llvm.ptr
    %351 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %352 = llvm.ptrtoint %351 : !llvm.ptr to i64
    %353 = llvm.add %352, %3 : i64
    %354 = llvm.call @_mlir_memref_to_llvm_alloc(%353) : (i64) -> !llvm.ptr
    %355 = llvm.ptrtoint %354 : !llvm.ptr to i64
    %356 = llvm.sub %3, %36 : i64
    %357 = llvm.add %355, %356 : i64
    %358 = llvm.urem %357, %3 : i64
    %359 = llvm.sub %357, %358 : i64
    %360 = llvm.inttoptr %359 : i64 to !llvm.ptr
    %361 = llvm.getelementptr %11[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %362 = llvm.ptrtoint %361 : !llvm.ptr to i64
    %363 = llvm.add %362, %3 : i64
    %364 = llvm.call @_mlir_memref_to_llvm_alloc(%363) : (i64) -> !llvm.ptr
    %365 = llvm.ptrtoint %364 : !llvm.ptr to i64
    %366 = llvm.sub %3, %36 : i64
    %367 = llvm.add %365, %366 : i64
    %368 = llvm.urem %367, %3 : i64
    %369 = llvm.sub %367, %368 : i64
    %370 = llvm.inttoptr %369 : i64 to !llvm.ptr
    %371 = llvm.getelementptr %11[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %372 = llvm.ptrtoint %371 : !llvm.ptr to i64
    %373 = llvm.add %372, %3 : i64
    %374 = llvm.call @_mlir_memref_to_llvm_alloc(%373) : (i64) -> !llvm.ptr
    %375 = llvm.ptrtoint %374 : !llvm.ptr to i64
    %376 = llvm.sub %3, %36 : i64
    %377 = llvm.add %375, %376 : i64
    %378 = llvm.urem %377, %3 : i64
    %379 = llvm.sub %377, %378 : i64
    %380 = llvm.inttoptr %379 : i64 to !llvm.ptr
    %381 = llvm.getelementptr %11[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %382 = llvm.ptrtoint %381 : !llvm.ptr to i64
    %383 = llvm.add %382, %3 : i64
    %384 = llvm.call @_mlir_memref_to_llvm_alloc(%383) : (i64) -> !llvm.ptr
    %385 = llvm.ptrtoint %384 : !llvm.ptr to i64
    %386 = llvm.sub %3, %36 : i64
    %387 = llvm.add %385, %386 : i64
    %388 = llvm.urem %387, %3 : i64
    %389 = llvm.sub %387, %388 : i64
    %390 = llvm.inttoptr %389 : i64 to !llvm.ptr
    %391 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %392 = llvm.ptrtoint %391 : !llvm.ptr to i64
    %393 = llvm.add %392, %3 : i64
    %394 = llvm.call @_mlir_memref_to_llvm_alloc(%393) : (i64) -> !llvm.ptr
    %395 = llvm.ptrtoint %394 : !llvm.ptr to i64
    %396 = llvm.sub %3, %36 : i64
    %397 = llvm.add %395, %396 : i64
    %398 = llvm.urem %397, %3 : i64
    %399 = llvm.sub %397, %398 : i64
    %400 = llvm.inttoptr %399 : i64 to !llvm.ptr
    %401 = llvm.getelementptr %11[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %402 = llvm.ptrtoint %401 : !llvm.ptr to i64
    %403 = llvm.add %402, %3 : i64
    %404 = llvm.call @_mlir_memref_to_llvm_alloc(%403) : (i64) -> !llvm.ptr
    %405 = llvm.ptrtoint %404 : !llvm.ptr to i64
    %406 = llvm.sub %3, %36 : i64
    %407 = llvm.add %405, %406 : i64
    %408 = llvm.urem %407, %3 : i64
    %409 = llvm.sub %407, %408 : i64
    %410 = llvm.inttoptr %409 : i64 to !llvm.ptr
    %411 = llvm.getelementptr %11[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %412 = llvm.ptrtoint %411 : !llvm.ptr to i64
    %413 = llvm.add %412, %3 : i64
    %414 = llvm.call @_mlir_memref_to_llvm_alloc(%413) : (i64) -> !llvm.ptr
    %415 = llvm.ptrtoint %414 : !llvm.ptr to i64
    %416 = llvm.sub %3, %36 : i64
    %417 = llvm.add %415, %416 : i64
    %418 = llvm.urem %417, %3 : i64
    %419 = llvm.sub %417, %418 : i64
    %420 = llvm.inttoptr %419 : i64 to !llvm.ptr
    %421 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %422 = llvm.ptrtoint %421 : !llvm.ptr to i64
    %423 = llvm.add %422, %3 : i64
    %424 = llvm.call @_mlir_memref_to_llvm_alloc(%423) : (i64) -> !llvm.ptr
    %425 = llvm.ptrtoint %424 : !llvm.ptr to i64
    %426 = llvm.sub %3, %36 : i64
    %427 = llvm.add %425, %426 : i64
    %428 = llvm.urem %427, %3 : i64
    %429 = llvm.sub %427, %428 : i64
    %430 = llvm.inttoptr %429 : i64 to !llvm.ptr
    %431 = llvm.getelementptr %11[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %432 = llvm.ptrtoint %431 : !llvm.ptr to i64
    %433 = llvm.call @_mlir_memref_to_llvm_alloc(%432) : (i64) -> !llvm.ptr
    %434 = llvm.insertvalue %433, %42[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %435 = llvm.insertvalue %433, %434[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %436 = llvm.insertvalue %14, %435[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %437 = llvm.insertvalue %37, %436[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %438 = llvm.insertvalue %36, %437[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %439 = llvm.mul %37, %36 : i64
    %440 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %441 = llvm.ptrtoint %440 : !llvm.ptr to i64
    %442 = llvm.mul %439, %441 : i64
    "llvm.intr.memcpy"(%433, %124, %442) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%118) : (!llvm.ptr) -> ()
    %443 = llvm.getelementptr %11[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %444 = llvm.ptrtoint %443 : !llvm.ptr to i64
    %445 = llvm.call @_mlir_memref_to_llvm_alloc(%444) : (i64) -> !llvm.ptr
    %446 = llvm.insertvalue %445, %42[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %447 = llvm.insertvalue %445, %446[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %448 = llvm.insertvalue %14, %447[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %449 = llvm.insertvalue %37, %448[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %450 = llvm.insertvalue %36, %449[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %451 = llvm.mul %37, %36 : i64
    %452 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %453 = llvm.ptrtoint %452 : !llvm.ptr to i64
    %454 = llvm.mul %451, %453 : i64
    "llvm.intr.memcpy"(%445, %175, %454) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%169) : (!llvm.ptr) -> ()
    %455 = llvm.getelementptr %11[4] : (!llvm.ptr) -> !llvm.ptr, i32
    %456 = llvm.ptrtoint %455 : !llvm.ptr to i64
    %457 = llvm.call @_mlir_memref_to_llvm_alloc(%456) : (i64) -> !llvm.ptr
    %458 = llvm.insertvalue %457, %42[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %459 = llvm.insertvalue %457, %458[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %460 = llvm.insertvalue %14, %459[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %461 = llvm.insertvalue %38, %460[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %462 = llvm.insertvalue %36, %461[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %463 = llvm.mul %38, %36 : i64
    %464 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %465 = llvm.ptrtoint %464 : !llvm.ptr to i64
    %466 = llvm.mul %463, %465 : i64
    "llvm.intr.memcpy"(%457, %212, %466) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%206) : (!llvm.ptr) -> ()
    %467 = llvm.getelementptr %11[4] : (!llvm.ptr) -> !llvm.ptr, i32
    %468 = llvm.ptrtoint %467 : !llvm.ptr to i64
    %469 = llvm.call @_mlir_memref_to_llvm_alloc(%468) : (i64) -> !llvm.ptr
    %470 = llvm.insertvalue %469, %42[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %471 = llvm.insertvalue %469, %470[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %472 = llvm.insertvalue %14, %471[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %473 = llvm.insertvalue %38, %472[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %474 = llvm.insertvalue %36, %473[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %475 = llvm.mul %38, %36 : i64
    %476 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %477 = llvm.ptrtoint %476 : !llvm.ptr to i64
    %478 = llvm.mul %475, %477 : i64
    "llvm.intr.memcpy"(%469, %226, %478) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%220) : (!llvm.ptr) -> ()
    llvm.br ^bb34(%18, %18, %450, %438, %201, %160, %202, %462, %474 : i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, i32, i32, i32, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)
  ^bb34(%479: i64, %480: i64, %481: !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, %482: !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, %483: i32, %484: i32, %485: i32, %486: !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, %487: !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>):  // 2 preds: ^bb33, ^bb137
    %488 = llvm.icmp "slt" %479, %27 : i64
    llvm.cond_br %488, ^bb35, ^bb138
  ^bb35:  // pred: ^bb34
    llvm.store %483, %240 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    llvm.store %484, %250 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %489 = llvm.add %480, %28 : i64
    %490 = llvm.extractvalue %486[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %491 = llvm.load %490 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %492 = llvm.getelementptr %11[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %493 = llvm.ptrtoint %492 : !llvm.ptr to i64
    %494 = llvm.add %493, %3 : i64
    %495 = llvm.call @_mlir_memref_to_llvm_alloc(%494) : (i64) -> !llvm.ptr
    %496 = llvm.ptrtoint %495 : !llvm.ptr to i64
    %497 = llvm.sub %3, %36 : i64
    %498 = llvm.add %496, %497 : i64
    %499 = llvm.urem %498, %3 : i64
    %500 = llvm.sub %498, %499 : i64
    %501 = llvm.inttoptr %500 : i64 to !llvm.ptr
    llvm.br ^bb36(%14 : i64)
  ^bb36(%502: i64):  // 2 preds: ^bb35, ^bb37
    %503 = llvm.icmp "slt" %502, %37 : i64
    llvm.cond_br %503, ^bb37, ^bb38
  ^bb37:  // pred: ^bb36
    %504 = llvm.extractvalue %481[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %505 = llvm.getelementptr inbounds %504[%502] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %506 = llvm.load %505 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %507 = llvm.extractvalue %482[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %508 = llvm.getelementptr inbounds %507[%502] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %509 = llvm.load %508 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %510 = llvm.add %506, %509 : i32
    %511 = llvm.getelementptr inbounds %260[%502] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %510, %511 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %512 = llvm.add %502, %36 : i64
    llvm.br ^bb36(%512 : i64)
  ^bb38:  // pred: ^bb36
    %513 = llvm.extractvalue %481[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.call @_mlir_memref_to_llvm_free(%513) : (!llvm.ptr) -> ()
    llvm.br ^bb39(%14 : i64)
  ^bb39(%514: i64):  // 2 preds: ^bb38, ^bb40
    %515 = llvm.icmp "slt" %514, %37 : i64
    llvm.cond_br %515, ^bb40, ^bb41
  ^bb40:  // pred: ^bb39
    %516 = llvm.load %490 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %517 = llvm.getelementptr inbounds %501[%514] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %516, %517 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %518 = llvm.add %514, %36 : i64
    llvm.br ^bb39(%518 : i64)
  ^bb41:  // pred: ^bb39
    llvm.br ^bb42(%14 : i64)
  ^bb42(%519: i64):  // 2 preds: ^bb41, ^bb43
    %520 = llvm.icmp "slt" %519, %37 : i64
    llvm.cond_br %520, ^bb43, ^bb44
  ^bb43:  // pred: ^bb42
    %521 = llvm.extractvalue %482[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %522 = llvm.getelementptr inbounds %521[%519] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %523 = llvm.load %522 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %524 = llvm.getelementptr inbounds %501[%519] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %525 = llvm.load %524 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %526 = llvm.shl %523, %525 : i32
    %527 = llvm.icmp "ult" %525, %26 : i32
    %528 = llvm.select %527, %526, %25 : i1, i32
    %529 = llvm.getelementptr inbounds %270[%519] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %528, %529 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %530 = llvm.add %519, %36 : i64
    llvm.br ^bb42(%530 : i64)
  ^bb44:  // pred: ^bb42
    %531 = llvm.sub %26, %491 : i32
    llvm.store %531, %280 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    llvm.br ^bb45(%14 : i64)
  ^bb45(%532: i64):  // 2 preds: ^bb44, ^bb46
    %533 = llvm.icmp "slt" %532, %37 : i64
    llvm.cond_br %533, ^bb46, ^bb47
  ^bb46:  // pred: ^bb45
    %534 = llvm.load %280 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %535 = llvm.getelementptr inbounds %501[%532] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %534, %535 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %536 = llvm.add %532, %36 : i64
    llvm.br ^bb45(%536 : i64)
  ^bb47:  // pred: ^bb45
    llvm.br ^bb48(%14 : i64)
  ^bb48(%537: i64):  // 2 preds: ^bb47, ^bb49
    %538 = llvm.icmp "slt" %537, %37 : i64
    llvm.cond_br %538, ^bb49, ^bb50
  ^bb49:  // pred: ^bb48
    %539 = llvm.extractvalue %482[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %540 = llvm.getelementptr inbounds %539[%537] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %541 = llvm.load %540 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %542 = llvm.getelementptr inbounds %501[%537] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %543 = llvm.load %542 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %544 = llvm.lshr %541, %543 : i32
    %545 = llvm.icmp "ult" %543, %26 : i32
    %546 = llvm.select %545, %544, %25 : i1, i32
    %547 = llvm.getelementptr inbounds %501[%537] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %546, %547 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %548 = llvm.add %537, %36 : i64
    llvm.br ^bb48(%548 : i64)
  ^bb50:  // pred: ^bb48
    %549 = llvm.extractvalue %482[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.call @_mlir_memref_to_llvm_free(%549) : (!llvm.ptr) -> ()
    llvm.br ^bb51(%14 : i64)
  ^bb51(%550: i64):  // 2 preds: ^bb50, ^bb52
    %551 = llvm.icmp "slt" %550, %37 : i64
    llvm.cond_br %551, ^bb52, ^bb53
  ^bb52:  // pred: ^bb51
    %552 = llvm.getelementptr inbounds %270[%550] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %553 = llvm.load %552 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %554 = llvm.getelementptr inbounds %501[%550] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %555 = llvm.load %554 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %556 = llvm.or %553, %555 : i32
    %557 = llvm.getelementptr inbounds %501[%550] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %556, %557 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %558 = llvm.add %550, %36 : i64
    llvm.br ^bb51(%558 : i64)
  ^bb53:  // pred: ^bb51
    llvm.br ^bb54(%14 : i64)
  ^bb54(%559: i64):  // 2 preds: ^bb53, ^bb55
    %560 = llvm.icmp "slt" %559, %37 : i64
    llvm.cond_br %560, ^bb55, ^bb56
  ^bb55:  // pred: ^bb54
    %561 = llvm.getelementptr inbounds %260[%559] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %562 = llvm.load %561 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %563 = llvm.getelementptr inbounds %501[%559] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %564 = llvm.load %563 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %565 = llvm.xor %562, %564 : i32
    %566 = llvm.getelementptr inbounds %290[%559] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %565, %566 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %567 = llvm.add %559, %36 : i64
    llvm.br ^bb54(%567 : i64)
  ^bb56:  // pred: ^bb54
    %568 = llvm.extractvalue %486[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %569 = llvm.getelementptr inbounds %568[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %570 = llvm.load %569 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    llvm.br ^bb57(%14 : i64)
  ^bb57(%571: i64):  // 2 preds: ^bb56, ^bb58
    %572 = llvm.icmp "slt" %571, %37 : i64
    llvm.cond_br %572, ^bb58, ^bb59
  ^bb58:  // pred: ^bb57
    %573 = llvm.getelementptr inbounds %260[%571] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %574 = llvm.load %573 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %575 = llvm.getelementptr inbounds %290[%571] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %576 = llvm.load %575 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %577 = llvm.add %574, %576 : i32
    %578 = llvm.getelementptr inbounds %300[%571] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %577, %578 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %579 = llvm.add %571, %36 : i64
    llvm.br ^bb57(%579 : i64)
  ^bb59:  // pred: ^bb57
    llvm.br ^bb60(%14 : i64)
  ^bb60(%580: i64):  // 2 preds: ^bb59, ^bb61
    %581 = llvm.icmp "slt" %580, %37 : i64
    llvm.cond_br %581, ^bb61, ^bb62
  ^bb61:  // pred: ^bb60
    %582 = llvm.getelementptr inbounds %568[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %583 = llvm.load %582 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %584 = llvm.getelementptr inbounds %501[%580] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %583, %584 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %585 = llvm.add %580, %36 : i64
    llvm.br ^bb60(%585 : i64)
  ^bb62:  // pred: ^bb60
    llvm.br ^bb63(%14 : i64)
  ^bb63(%586: i64):  // 2 preds: ^bb62, ^bb64
    %587 = llvm.icmp "slt" %586, %37 : i64
    llvm.cond_br %587, ^bb64, ^bb65
  ^bb64:  // pred: ^bb63
    %588 = llvm.getelementptr inbounds %290[%586] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %589 = llvm.load %588 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %590 = llvm.getelementptr inbounds %501[%586] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %591 = llvm.load %590 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %592 = llvm.shl %589, %591 : i32
    %593 = llvm.icmp "ult" %591, %26 : i32
    %594 = llvm.select %593, %592, %25 : i1, i32
    %595 = llvm.getelementptr inbounds %310[%586] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %594, %595 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %596 = llvm.add %586, %36 : i64
    llvm.br ^bb63(%596 : i64)
  ^bb65:  // pred: ^bb63
    %597 = llvm.sub %26, %570 : i32
    llvm.store %597, %320 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    llvm.br ^bb66(%14 : i64)
  ^bb66(%598: i64):  // 2 preds: ^bb65, ^bb67
    %599 = llvm.icmp "slt" %598, %37 : i64
    llvm.cond_br %599, ^bb67, ^bb68
  ^bb67:  // pred: ^bb66
    %600 = llvm.load %320 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %601 = llvm.getelementptr inbounds %501[%598] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %600, %601 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %602 = llvm.add %598, %36 : i64
    llvm.br ^bb66(%602 : i64)
  ^bb68:  // pred: ^bb66
    llvm.br ^bb69(%14 : i64)
  ^bb69(%603: i64):  // 2 preds: ^bb68, ^bb70
    %604 = llvm.icmp "slt" %603, %37 : i64
    llvm.cond_br %604, ^bb70, ^bb71
  ^bb70:  // pred: ^bb69
    %605 = llvm.getelementptr inbounds %290[%603] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %606 = llvm.load %605 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %607 = llvm.getelementptr inbounds %501[%603] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %608 = llvm.load %607 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %609 = llvm.lshr %606, %608 : i32
    %610 = llvm.icmp "ult" %608, %26 : i32
    %611 = llvm.select %610, %609, %25 : i1, i32
    %612 = llvm.getelementptr inbounds %501[%603] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %611, %612 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %613 = llvm.add %603, %36 : i64
    llvm.br ^bb69(%613 : i64)
  ^bb71:  // pred: ^bb69
    llvm.br ^bb72(%14 : i64)
  ^bb72(%614: i64):  // 2 preds: ^bb71, ^bb73
    %615 = llvm.icmp "slt" %614, %37 : i64
    llvm.cond_br %615, ^bb73, ^bb74
  ^bb73:  // pred: ^bb72
    %616 = llvm.getelementptr inbounds %310[%614] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %617 = llvm.load %616 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %618 = llvm.getelementptr inbounds %501[%614] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %619 = llvm.load %618 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %620 = llvm.or %617, %619 : i32
    %621 = llvm.getelementptr inbounds %501[%614] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %620, %621 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %622 = llvm.add %614, %36 : i64
    llvm.br ^bb72(%622 : i64)
  ^bb74:  // pred: ^bb72
    llvm.br ^bb75(%14 : i64)
  ^bb75(%623: i64):  // 2 preds: ^bb74, ^bb76
    %624 = llvm.icmp "slt" %623, %37 : i64
    llvm.cond_br %624, ^bb76, ^bb77
  ^bb76:  // pred: ^bb75
    %625 = llvm.getelementptr inbounds %300[%623] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %626 = llvm.load %625 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %627 = llvm.getelementptr inbounds %501[%623] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %628 = llvm.load %627 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %629 = llvm.xor %626, %628 : i32
    %630 = llvm.getelementptr inbounds %330[%623] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %629, %630 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %631 = llvm.add %623, %36 : i64
    llvm.br ^bb75(%631 : i64)
  ^bb77:  // pred: ^bb75
    %632 = llvm.extractvalue %486[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %633 = llvm.getelementptr inbounds %632[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %634 = llvm.load %633 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    llvm.br ^bb78(%14 : i64)
  ^bb78(%635: i64):  // 2 preds: ^bb77, ^bb79
    %636 = llvm.icmp "slt" %635, %37 : i64
    llvm.cond_br %636, ^bb79, ^bb80
  ^bb79:  // pred: ^bb78
    %637 = llvm.getelementptr inbounds %300[%635] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %638 = llvm.load %637 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %639 = llvm.getelementptr inbounds %330[%635] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %640 = llvm.load %639 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %641 = llvm.add %638, %640 : i32
    %642 = llvm.getelementptr inbounds %340[%635] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %641, %642 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %643 = llvm.add %635, %36 : i64
    llvm.br ^bb78(%643 : i64)
  ^bb80:  // pred: ^bb78
    llvm.br ^bb81(%14 : i64)
  ^bb81(%644: i64):  // 2 preds: ^bb80, ^bb82
    %645 = llvm.icmp "slt" %644, %37 : i64
    llvm.cond_br %645, ^bb82, ^bb83
  ^bb82:  // pred: ^bb81
    %646 = llvm.getelementptr inbounds %632[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %647 = llvm.load %646 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %648 = llvm.getelementptr inbounds %501[%644] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %647, %648 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %649 = llvm.add %644, %36 : i64
    llvm.br ^bb81(%649 : i64)
  ^bb83:  // pred: ^bb81
    llvm.br ^bb84(%14 : i64)
  ^bb84(%650: i64):  // 2 preds: ^bb83, ^bb85
    %651 = llvm.icmp "slt" %650, %37 : i64
    llvm.cond_br %651, ^bb85, ^bb86
  ^bb85:  // pred: ^bb84
    %652 = llvm.getelementptr inbounds %330[%650] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %653 = llvm.load %652 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %654 = llvm.getelementptr inbounds %501[%650] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %655 = llvm.load %654 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %656 = llvm.shl %653, %655 : i32
    %657 = llvm.icmp "ult" %655, %26 : i32
    %658 = llvm.select %657, %656, %25 : i1, i32
    %659 = llvm.getelementptr inbounds %350[%650] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %658, %659 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %660 = llvm.add %650, %36 : i64
    llvm.br ^bb84(%660 : i64)
  ^bb86:  // pred: ^bb84
    %661 = llvm.sub %26, %634 : i32
    llvm.store %661, %360 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    llvm.br ^bb87(%14 : i64)
  ^bb87(%662: i64):  // 2 preds: ^bb86, ^bb88
    %663 = llvm.icmp "slt" %662, %37 : i64
    llvm.cond_br %663, ^bb88, ^bb89
  ^bb88:  // pred: ^bb87
    %664 = llvm.load %360 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %665 = llvm.getelementptr inbounds %501[%662] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %664, %665 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %666 = llvm.add %662, %36 : i64
    llvm.br ^bb87(%666 : i64)
  ^bb89:  // pred: ^bb87
    llvm.br ^bb90(%14 : i64)
  ^bb90(%667: i64):  // 2 preds: ^bb89, ^bb91
    %668 = llvm.icmp "slt" %667, %37 : i64
    llvm.cond_br %668, ^bb91, ^bb92
  ^bb91:  // pred: ^bb90
    %669 = llvm.getelementptr inbounds %330[%667] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %670 = llvm.load %669 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %671 = llvm.getelementptr inbounds %501[%667] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %672 = llvm.load %671 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %673 = llvm.lshr %670, %672 : i32
    %674 = llvm.icmp "ult" %672, %26 : i32
    %675 = llvm.select %674, %673, %25 : i1, i32
    %676 = llvm.getelementptr inbounds %501[%667] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %675, %676 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %677 = llvm.add %667, %36 : i64
    llvm.br ^bb90(%677 : i64)
  ^bb92:  // pred: ^bb90
    llvm.br ^bb93(%14 : i64)
  ^bb93(%678: i64):  // 2 preds: ^bb92, ^bb94
    %679 = llvm.icmp "slt" %678, %37 : i64
    llvm.cond_br %679, ^bb94, ^bb95
  ^bb94:  // pred: ^bb93
    %680 = llvm.getelementptr inbounds %350[%678] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %681 = llvm.load %680 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %682 = llvm.getelementptr inbounds %501[%678] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %683 = llvm.load %682 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %684 = llvm.or %681, %683 : i32
    %685 = llvm.getelementptr inbounds %501[%678] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %684, %685 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %686 = llvm.add %678, %36 : i64
    llvm.br ^bb93(%686 : i64)
  ^bb95:  // pred: ^bb93
    llvm.br ^bb96(%14 : i64)
  ^bb96(%687: i64):  // 2 preds: ^bb95, ^bb97
    %688 = llvm.icmp "slt" %687, %37 : i64
    llvm.cond_br %688, ^bb97, ^bb98
  ^bb97:  // pred: ^bb96
    %689 = llvm.getelementptr inbounds %340[%687] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %690 = llvm.load %689 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %691 = llvm.getelementptr inbounds %501[%687] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %692 = llvm.load %691 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %693 = llvm.xor %690, %692 : i32
    %694 = llvm.getelementptr inbounds %370[%687] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %693, %694 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %695 = llvm.add %687, %36 : i64
    llvm.br ^bb96(%695 : i64)
  ^bb98:  // pred: ^bb96
    %696 = llvm.extractvalue %486[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %697 = llvm.getelementptr inbounds %696[3] : (!llvm.ptr) -> !llvm.ptr, i32
    %698 = llvm.load %697 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    llvm.br ^bb99(%14 : i64)
  ^bb99(%699: i64):  // 2 preds: ^bb98, ^bb100
    %700 = llvm.icmp "slt" %699, %37 : i64
    llvm.cond_br %700, ^bb100, ^bb101
  ^bb100:  // pred: ^bb99
    %701 = llvm.getelementptr inbounds %340[%699] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %702 = llvm.load %701 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %703 = llvm.getelementptr inbounds %370[%699] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %704 = llvm.load %703 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %705 = llvm.add %702, %704 : i32
    %706 = llvm.getelementptr inbounds %380[%699] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %705, %706 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %707 = llvm.add %699, %36 : i64
    llvm.br ^bb99(%707 : i64)
  ^bb101:  // pred: ^bb99
    llvm.br ^bb102(%14 : i64)
  ^bb102(%708: i64):  // 2 preds: ^bb101, ^bb103
    %709 = llvm.icmp "slt" %708, %37 : i64
    llvm.cond_br %709, ^bb103, ^bb104
  ^bb103:  // pred: ^bb102
    %710 = llvm.getelementptr inbounds %696[3] : (!llvm.ptr) -> !llvm.ptr, i32
    %711 = llvm.load %710 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %712 = llvm.getelementptr inbounds %501[%708] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %711, %712 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %713 = llvm.add %708, %36 : i64
    llvm.br ^bb102(%713 : i64)
  ^bb104:  // pred: ^bb102
    llvm.br ^bb105(%14 : i64)
  ^bb105(%714: i64):  // 2 preds: ^bb104, ^bb106
    %715 = llvm.icmp "slt" %714, %37 : i64
    llvm.cond_br %715, ^bb106, ^bb107
  ^bb106:  // pred: ^bb105
    %716 = llvm.getelementptr inbounds %370[%714] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %717 = llvm.load %716 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %718 = llvm.getelementptr inbounds %501[%714] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %719 = llvm.load %718 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %720 = llvm.shl %717, %719 : i32
    %721 = llvm.icmp "ult" %719, %26 : i32
    %722 = llvm.select %721, %720, %25 : i1, i32
    %723 = llvm.getelementptr inbounds %390[%714] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %722, %723 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %724 = llvm.add %714, %36 : i64
    llvm.br ^bb105(%724 : i64)
  ^bb107:  // pred: ^bb105
    %725 = llvm.sub %26, %698 : i32
    llvm.store %725, %400 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    llvm.br ^bb108(%14 : i64)
  ^bb108(%726: i64):  // 2 preds: ^bb107, ^bb109
    %727 = llvm.icmp "slt" %726, %37 : i64
    llvm.cond_br %727, ^bb109, ^bb110
  ^bb109:  // pred: ^bb108
    %728 = llvm.load %400 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %729 = llvm.getelementptr inbounds %501[%726] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %728, %729 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %730 = llvm.add %726, %36 : i64
    llvm.br ^bb108(%730 : i64)
  ^bb110:  // pred: ^bb108
    llvm.br ^bb111(%14 : i64)
  ^bb111(%731: i64):  // 2 preds: ^bb110, ^bb112
    %732 = llvm.icmp "slt" %731, %37 : i64
    llvm.cond_br %732, ^bb112, ^bb113
  ^bb112:  // pred: ^bb111
    %733 = llvm.getelementptr inbounds %370[%731] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %734 = llvm.load %733 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %735 = llvm.getelementptr inbounds %501[%731] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %736 = llvm.load %735 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %737 = llvm.lshr %734, %736 : i32
    %738 = llvm.icmp "ult" %736, %26 : i32
    %739 = llvm.select %738, %737, %25 : i1, i32
    %740 = llvm.getelementptr inbounds %501[%731] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %739, %740 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %741 = llvm.add %731, %36 : i64
    llvm.br ^bb111(%741 : i64)
  ^bb113:  // pred: ^bb111
    llvm.br ^bb114(%14 : i64)
  ^bb114(%742: i64):  // 2 preds: ^bb113, ^bb115
    %743 = llvm.icmp "slt" %742, %37 : i64
    llvm.cond_br %743, ^bb115, ^bb116
  ^bb115:  // pred: ^bb114
    %744 = llvm.getelementptr inbounds %390[%742] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %745 = llvm.load %744 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %746 = llvm.getelementptr inbounds %501[%742] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %747 = llvm.load %746 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %748 = llvm.or %745, %747 : i32
    %749 = llvm.getelementptr inbounds %501[%742] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %748, %749 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %750 = llvm.add %742, %36 : i64
    llvm.br ^bb114(%750 : i64)
  ^bb116:  // pred: ^bb114
    llvm.br ^bb117(%14 : i64)
  ^bb117(%751: i64):  // 2 preds: ^bb116, ^bb118
    %752 = llvm.icmp "slt" %751, %37 : i64
    llvm.cond_br %752, ^bb118, ^bb119
  ^bb118:  // pred: ^bb117
    %753 = llvm.getelementptr inbounds %380[%751] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %754 = llvm.load %753 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %755 = llvm.getelementptr inbounds %501[%751] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %756 = llvm.load %755 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %757 = llvm.xor %754, %756 : i32
    %758 = llvm.getelementptr inbounds %410[%751] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %757, %758 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %759 = llvm.add %751, %36 : i64
    llvm.br ^bb117(%759 : i64)
  ^bb119:  // pred: ^bb117
    llvm.br ^bb120(%14 : i64)
  ^bb120(%760: i64):  // 2 preds: ^bb119, ^bb121
    %761 = llvm.icmp "slt" %760, %37 : i64
    llvm.cond_br %761, ^bb121, ^bb122
  ^bb121:  // pred: ^bb120
    %762 = llvm.load %240 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %763 = llvm.getelementptr inbounds %501[%760] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %762, %763 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %764 = llvm.add %760, %36 : i64
    llvm.br ^bb120(%764 : i64)
  ^bb122:  // pred: ^bb120
    %765 = llvm.getelementptr %11[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %766 = llvm.ptrtoint %765 : !llvm.ptr to i64
    %767 = llvm.add %766, %3 : i64
    %768 = llvm.call @_mlir_memref_to_llvm_alloc(%767) : (i64) -> !llvm.ptr
    %769 = llvm.ptrtoint %768 : !llvm.ptr to i64
    %770 = llvm.sub %3, %36 : i64
    %771 = llvm.add %769, %770 : i64
    %772 = llvm.urem %771, %3 : i64
    %773 = llvm.sub %771, %772 : i64
    %774 = llvm.inttoptr %773 : i64 to !llvm.ptr
    llvm.br ^bb123(%14 : i64)
  ^bb123(%775: i64):  // 2 preds: ^bb122, ^bb124
    %776 = llvm.icmp "slt" %775, %37 : i64
    llvm.cond_br %776, ^bb124, ^bb125
  ^bb124:  // pred: ^bb123
    %777 = llvm.getelementptr inbounds %380[%775] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %778 = llvm.load %777 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %779 = llvm.getelementptr inbounds %501[%775] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %780 = llvm.load %779 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %781 = llvm.add %778, %780 : i32
    %782 = llvm.getelementptr inbounds %774[%775] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %781, %782 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %783 = llvm.add %775, %36 : i64
    llvm.br ^bb123(%783 : i64)
  ^bb125:  // pred: ^bb123
    llvm.br ^bb126(%14 : i64)
  ^bb126(%784: i64):  // 2 preds: ^bb125, ^bb127
    %785 = llvm.icmp "slt" %784, %37 : i64
    llvm.cond_br %785, ^bb127, ^bb128
  ^bb127:  // pred: ^bb126
    %786 = llvm.load %250 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %787 = llvm.getelementptr inbounds %501[%784] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %786, %787 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %788 = llvm.add %784, %36 : i64
    llvm.br ^bb126(%788 : i64)
  ^bb128:  // pred: ^bb126
    llvm.br ^bb129(%14 : i64)
  ^bb129(%789: i64):  // 2 preds: ^bb128, ^bb130
    %790 = llvm.icmp "slt" %789, %37 : i64
    llvm.cond_br %790, ^bb130, ^bb131
  ^bb130:  // pred: ^bb129
    %791 = llvm.getelementptr inbounds %410[%789] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %792 = llvm.load %791 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %793 = llvm.getelementptr inbounds %501[%789] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %794 = llvm.load %793 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %795 = llvm.add %792, %794 : i32
    %796 = llvm.getelementptr inbounds %420[%789] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %795, %796 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %797 = llvm.add %789, %36 : i64
    llvm.br ^bb129(%797 : i64)
  ^bb131:  // pred: ^bb129
    %798 = llvm.trunc %489 : i64 to i32
    llvm.store %798, %430 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    llvm.br ^bb132(%14 : i64)
  ^bb132(%799: i64):  // 2 preds: ^bb131, ^bb133
    %800 = llvm.icmp "slt" %799, %37 : i64
    llvm.cond_br %800, ^bb133, ^bb134
  ^bb133:  // pred: ^bb132
    %801 = llvm.load %430 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %802 = llvm.getelementptr inbounds %501[%799] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %801, %802 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %803 = llvm.add %799, %36 : i64
    llvm.br ^bb132(%803 : i64)
  ^bb134:  // pred: ^bb132
    llvm.br ^bb135(%14 : i64)
  ^bb135(%804: i64):  // 2 preds: ^bb134, ^bb136
    %805 = llvm.icmp "slt" %804, %37 : i64
    llvm.cond_br %805, ^bb136, ^bb137
  ^bb136:  // pred: ^bb135
    %806 = llvm.getelementptr inbounds %420[%804] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %807 = llvm.load %806 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %808 = llvm.getelementptr inbounds %501[%804] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %809 = llvm.load %808 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %810 = llvm.add %807, %809 : i32
    %811 = llvm.getelementptr inbounds %501[%804] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %810, %811 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %812 = llvm.add %804, %36 : i64
    llvm.br ^bb135(%812 : i64)
  ^bb137:  // pred: ^bb135
    %813 = llvm.getelementptr %11[4] : (!llvm.ptr) -> !llvm.ptr, i32
    %814 = llvm.ptrtoint %813 : !llvm.ptr to i64
    %815 = llvm.add %814, %3 : i64
    %816 = llvm.call @_mlir_memref_to_llvm_alloc(%815) : (i64) -> !llvm.ptr
    %817 = llvm.ptrtoint %816 : !llvm.ptr to i64
    %818 = llvm.sub %3, %36 : i64
    %819 = llvm.add %817, %818 : i64
    %820 = llvm.urem %819, %3 : i64
    %821 = llvm.sub %819, %820 : i64
    %822 = llvm.inttoptr %821 : i64 to !llvm.ptr
    %823 = llvm.extractvalue %487[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %824 = llvm.mul %823, %36 : i64
    %825 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %826 = llvm.ptrtoint %825 : !llvm.ptr to i64
    %827 = llvm.mul %824, %826 : i64
    %828 = llvm.extractvalue %487[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %829 = llvm.extractvalue %487[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %830 = llvm.getelementptr inbounds %828[%829] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    "llvm.intr.memcpy"(%822, %830, %827) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %831 = llvm.extractvalue %487[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.call @_mlir_memref_to_llvm_free(%831) : (!llvm.ptr) -> ()
    %832 = llvm.getelementptr %11[4] : (!llvm.ptr) -> !llvm.ptr, i32
    %833 = llvm.ptrtoint %832 : !llvm.ptr to i64
    %834 = llvm.add %833, %3 : i64
    %835 = llvm.call @_mlir_memref_to_llvm_alloc(%834) : (i64) -> !llvm.ptr
    %836 = llvm.ptrtoint %835 : !llvm.ptr to i64
    %837 = llvm.sub %3, %36 : i64
    %838 = llvm.add %836, %837 : i64
    %839 = llvm.urem %838, %3 : i64
    %840 = llvm.sub %838, %839 : i64
    %841 = llvm.inttoptr %840 : i64 to !llvm.ptr
    %842 = llvm.extractvalue %486[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %843 = llvm.mul %842, %36 : i64
    %844 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %845 = llvm.ptrtoint %844 : !llvm.ptr to i64
    %846 = llvm.mul %843, %845 : i64
    %847 = llvm.extractvalue %486[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %848 = llvm.extractvalue %486[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %849 = llvm.getelementptr inbounds %847[%848] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    "llvm.intr.memcpy"(%841, %849, %846) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %850 = llvm.extractvalue %486[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.call @_mlir_memref_to_llvm_free(%850) : (!llvm.ptr) -> ()
    %851 = llvm.getelementptr %11[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %852 = llvm.ptrtoint %851 : !llvm.ptr to i64
    %853 = llvm.call @_mlir_memref_to_llvm_alloc(%852) : (i64) -> !llvm.ptr
    %854 = llvm.insertvalue %853, %42[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %855 = llvm.insertvalue %853, %854[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %856 = llvm.insertvalue %14, %855[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %857 = llvm.insertvalue %37, %856[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %858 = llvm.insertvalue %36, %857[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %859 = llvm.mul %37, %36 : i64
    %860 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %861 = llvm.ptrtoint %860 : !llvm.ptr to i64
    %862 = llvm.mul %859, %861 : i64
    "llvm.intr.memcpy"(%853, %501, %862) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%495) : (!llvm.ptr) -> ()
    %863 = llvm.getelementptr %11[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %864 = llvm.ptrtoint %863 : !llvm.ptr to i64
    %865 = llvm.call @_mlir_memref_to_llvm_alloc(%864) : (i64) -> !llvm.ptr
    %866 = llvm.insertvalue %865, %42[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %867 = llvm.insertvalue %865, %866[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %868 = llvm.insertvalue %14, %867[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %869 = llvm.insertvalue %37, %868[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %870 = llvm.insertvalue %36, %869[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %871 = llvm.mul %37, %36 : i64
    %872 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %873 = llvm.ptrtoint %872 : !llvm.ptr to i64
    %874 = llvm.mul %871, %873 : i64
    "llvm.intr.memcpy"(%865, %774, %874) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%768) : (!llvm.ptr) -> ()
    %875 = llvm.getelementptr %11[4] : (!llvm.ptr) -> !llvm.ptr, i32
    %876 = llvm.ptrtoint %875 : !llvm.ptr to i64
    %877 = llvm.call @_mlir_memref_to_llvm_alloc(%876) : (i64) -> !llvm.ptr
    %878 = llvm.insertvalue %877, %42[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %879 = llvm.insertvalue %877, %878[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %880 = llvm.insertvalue %14, %879[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %881 = llvm.insertvalue %38, %880[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %882 = llvm.insertvalue %36, %881[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %883 = llvm.mul %38, %36 : i64
    %884 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %885 = llvm.ptrtoint %884 : !llvm.ptr to i64
    %886 = llvm.mul %883, %885 : i64
    "llvm.intr.memcpy"(%877, %822, %886) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%816) : (!llvm.ptr) -> ()
    %887 = llvm.getelementptr %11[4] : (!llvm.ptr) -> !llvm.ptr, i32
    %888 = llvm.ptrtoint %887 : !llvm.ptr to i64
    %889 = llvm.call @_mlir_memref_to_llvm_alloc(%888) : (i64) -> !llvm.ptr
    %890 = llvm.insertvalue %889, %42[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %891 = llvm.insertvalue %889, %890[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %892 = llvm.insertvalue %14, %891[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %893 = llvm.insertvalue %38, %892[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %894 = llvm.insertvalue %36, %893[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %895 = llvm.mul %38, %36 : i64
    %896 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %897 = llvm.ptrtoint %896 : !llvm.ptr to i64
    %898 = llvm.mul %895, %897 : i64
    "llvm.intr.memcpy"(%889, %841, %898) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%835) : (!llvm.ptr) -> ()
    %899 = llvm.add %479, %28 : i64
    llvm.br ^bb34(%899, %489, %870, %858, %484, %485, %483, %882, %894 : i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, i32, i32, i32, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)
  ^bb138:  // pred: ^bb34
    %900 = llvm.extractvalue %487[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.call @_mlir_memref_to_llvm_free(%900) : (!llvm.ptr) -> ()
    %901 = llvm.extractvalue %486[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.call @_mlir_memref_to_llvm_free(%901) : (!llvm.ptr) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%424) : (!llvm.ptr) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%414) : (!llvm.ptr) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%404) : (!llvm.ptr) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%394) : (!llvm.ptr) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%384) : (!llvm.ptr) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%374) : (!llvm.ptr) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%364) : (!llvm.ptr) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%354) : (!llvm.ptr) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%344) : (!llvm.ptr) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%334) : (!llvm.ptr) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%324) : (!llvm.ptr) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%314) : (!llvm.ptr) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%304) : (!llvm.ptr) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%294) : (!llvm.ptr) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%284) : (!llvm.ptr) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%274) : (!llvm.ptr) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%264) : (!llvm.ptr) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%254) : (!llvm.ptr) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%244) : (!llvm.ptr) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%234) : (!llvm.ptr) -> ()
    %902 = llvm.extractvalue %481[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %903 = llvm.extractvalue %482[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %904 = llvm.getelementptr %11[4] : (!llvm.ptr) -> !llvm.ptr, i32
    %905 = llvm.ptrtoint %904 : !llvm.ptr to i64
    %906 = llvm.add %905, %3 : i64
    %907 = llvm.call @_mlir_memref_to_llvm_alloc(%906) : (i64) -> !llvm.ptr
    %908 = llvm.ptrtoint %907 : !llvm.ptr to i64
    %909 = llvm.sub %3, %36 : i64
    %910 = llvm.add %908, %909 : i64
    %911 = llvm.urem %910, %3 : i64
    %912 = llvm.sub %910, %911 : i64
    %913 = llvm.inttoptr %912 : i64 to !llvm.ptr
    llvm.br ^bb139(%14 : i64)
  ^bb139(%914: i64):  // 2 preds: ^bb138, ^bb147
    %915 = llvm.icmp "slt" %914, %37 : i64
    llvm.cond_br %915, ^bb140, ^bb148
  ^bb140:  // pred: ^bb139
    llvm.br ^bb141(%14 : i64)
  ^bb141(%916: i64):  // 2 preds: ^bb140, ^bb146
    %917 = llvm.icmp "slt" %916, %37 : i64
    llvm.cond_br %917, ^bb142, ^bb147
  ^bb142:  // pred: ^bb141
    %918 = llvm.icmp "ult" %916, %36 : i64
    llvm.cond_br %918, ^bb143, ^bb144
  ^bb143:  // pred: ^bb142
    %919 = llvm.add %914, %916 : i64
    %920 = llvm.getelementptr inbounds %902[%919] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %921 = llvm.load %920 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    llvm.br ^bb145(%921 : i32)
  ^bb144:  // pred: ^bb142
    %922 = llvm.sub %916, %36 : i64
    %923 = llvm.add %914, %922 : i64
    %924 = llvm.getelementptr inbounds %903[%923] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %925 = llvm.load %924 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    llvm.br ^bb145(%925 : i32)
  ^bb145(%926: i32):  // 2 preds: ^bb143, ^bb144
    llvm.br ^bb146
  ^bb146:  // pred: ^bb145
    %927 = llvm.mul %914, %37 : i64
    %928 = llvm.add %927, %916 : i64
    %929 = llvm.getelementptr inbounds %913[%928] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %926, %929 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %930 = llvm.add %916, %36 : i64
    llvm.br ^bb141(%930 : i64)
  ^bb147:  // pred: ^bb141
    %931 = llvm.add %914, %36 : i64
    llvm.br ^bb139(%931 : i64)
  ^bb148:  // pred: ^bb139
    %932 = llvm.extractvalue %481[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.call @_mlir_memref_to_llvm_free(%932) : (!llvm.ptr) -> ()
    %933 = llvm.extractvalue %482[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.call @_mlir_memref_to_llvm_free(%933) : (!llvm.ptr) -> ()
    %934 = llvm.insertvalue %907, %42[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %935 = llvm.insertvalue %913, %934[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %936 = llvm.insertvalue %14, %935[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %937 = llvm.insertvalue %37, %936[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %938 = llvm.insertvalue %36, %937[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %939 = llvm.getelementptr inbounds %913[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %940 = llvm.load %939 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %941 = llvm.getelementptr inbounds %913[3] : (!llvm.ptr) -> !llvm.ptr, i32
    %942 = llvm.load %941 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %943 = llvm.getelementptr %11[2] : (!llvm.ptr) -> !llvm.ptr, i64
    %944 = llvm.ptrtoint %943 : !llvm.ptr to i64
    %945 = llvm.add %944, %3 : i64
    %946 = llvm.call @_mlir_memref_to_llvm_alloc(%945) : (i64) -> !llvm.ptr
    %947 = llvm.ptrtoint %946 : !llvm.ptr to i64
    %948 = llvm.sub %3, %36 : i64
    %949 = llvm.add %947, %948 : i64
    %950 = llvm.urem %949, %3 : i64
    %951 = llvm.sub %949, %950 : i64
    %952 = llvm.inttoptr %951 : i64 to !llvm.ptr
    %953 = llvm.getelementptr %11[2] : (!llvm.ptr) -> !llvm.ptr, i64
    %954 = llvm.ptrtoint %953 : !llvm.ptr to i64
    %955 = llvm.add %954, %3 : i64
    %956 = llvm.call @_mlir_memref_to_llvm_alloc(%955) : (i64) -> !llvm.ptr
    %957 = llvm.ptrtoint %956 : !llvm.ptr to i64
    %958 = llvm.sub %3, %36 : i64
    %959 = llvm.add %957, %958 : i64
    %960 = llvm.urem %959, %3 : i64
    %961 = llvm.sub %959, %960 : i64
    %962 = llvm.inttoptr %961 : i64 to !llvm.ptr
    llvm.br ^bb149(%14 : i64)
  ^bb149(%963: i64):  // 2 preds: ^bb148, ^bb150
    %964 = llvm.icmp "slt" %963, %37 : i64
    llvm.cond_br %964, ^bb150, ^bb151
  ^bb150:  // pred: ^bb149
    %965 = llvm.getelementptr inbounds %962[%963] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %963, %965 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %966 = llvm.add %963, %36 : i64
    llvm.br ^bb149(%966 : i64)
  ^bb151:  // pred: ^bb149
    llvm.br ^bb152(%14 : i64)
  ^bb152(%967: i64):  // 2 preds: ^bb151, ^bb153
    %968 = llvm.icmp "slt" %967, %37 : i64
    llvm.cond_br %968, ^bb153, ^bb154
  ^bb153:  // pred: ^bb152
    %969 = llvm.load %7 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %970 = llvm.getelementptr inbounds %952[%967] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %969, %970 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %971 = llvm.add %967, %36 : i64
    llvm.br ^bb152(%971 : i64)
  ^bb154:  // pred: ^bb152
    %972 = llvm.getelementptr %11[2] : (!llvm.ptr) -> !llvm.ptr, i64
    %973 = llvm.ptrtoint %972 : !llvm.ptr to i64
    %974 = llvm.add %973, %3 : i64
    %975 = llvm.call @_mlir_memref_to_llvm_alloc(%974) : (i64) -> !llvm.ptr
    %976 = llvm.ptrtoint %975 : !llvm.ptr to i64
    %977 = llvm.sub %3, %36 : i64
    %978 = llvm.add %976, %977 : i64
    %979 = llvm.urem %978, %3 : i64
    %980 = llvm.sub %978, %979 : i64
    %981 = llvm.inttoptr %980 : i64 to !llvm.ptr
    llvm.br ^bb155(%14 : i64)
  ^bb155(%982: i64):  // 2 preds: ^bb154, ^bb156
    %983 = llvm.icmp "slt" %982, %37 : i64
    llvm.cond_br %983, ^bb156, ^bb157
  ^bb156:  // pred: ^bb155
    %984 = llvm.getelementptr inbounds %952[%982] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %985 = llvm.load %984 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %986 = llvm.getelementptr inbounds %962[%982] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %987 = llvm.load %986 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %988 = llvm.mul %985, %987 : i64
    %989 = llvm.getelementptr inbounds %981[%982] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %988, %989 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %990 = llvm.add %982, %36 : i64
    llvm.br ^bb155(%990 : i64)
  ^bb157:  // pred: ^bb155
    llvm.call @_mlir_memref_to_llvm_free(%956) : (!llvm.ptr) -> ()
    llvm.br ^bb158(%14 : i64)
  ^bb158(%991: i64):  // 2 preds: ^bb157, ^bb159
    %992 = llvm.icmp "slt" %991, %37 : i64
    llvm.cond_br %992, ^bb159, ^bb160
  ^bb159:  // pred: ^bb158
    %993 = llvm.load %4 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %994 = llvm.getelementptr inbounds %952[%991] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %993, %994 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %995 = llvm.add %991, %36 : i64
    llvm.br ^bb158(%995 : i64)
  ^bb160:  // pred: ^bb158
    llvm.br ^bb161(%14 : i64)
  ^bb161(%996: i64):  // 2 preds: ^bb160, ^bb162
    %997 = llvm.icmp "slt" %996, %37 : i64
    llvm.cond_br %997, ^bb162, ^bb163
  ^bb162:  // pred: ^bb161
    %998 = llvm.getelementptr inbounds %981[%996] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %999 = llvm.load %998 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %1000 = llvm.getelementptr inbounds %952[%996] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %1001 = llvm.load %1000 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %1002 = llvm.lshr %999, %1001 : i64
    %1003 = llvm.icmp "ult" %1001, %29 : i64
    %1004 = llvm.select %1003, %1002, %18 : i1, i64
    %1005 = llvm.getelementptr inbounds %952[%996] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %1004, %1005 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %1006 = llvm.add %996, %36 : i64
    llvm.br ^bb161(%1006 : i64)
  ^bb163:  // pred: ^bb161
    %1007 = llvm.getelementptr %11[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %1008 = llvm.ptrtoint %1007 : !llvm.ptr to i64
    %1009 = llvm.add %1008, %3 : i64
    %1010 = llvm.call @_mlir_memref_to_llvm_alloc(%1009) : (i64) -> !llvm.ptr
    %1011 = llvm.ptrtoint %1010 : !llvm.ptr to i64
    %1012 = llvm.sub %3, %36 : i64
    %1013 = llvm.add %1011, %1012 : i64
    %1014 = llvm.urem %1013, %3 : i64
    %1015 = llvm.sub %1013, %1014 : i64
    %1016 = llvm.inttoptr %1015 : i64 to !llvm.ptr
    %1017 = llvm.getelementptr %11[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %1018 = llvm.ptrtoint %1017 : !llvm.ptr to i64
    %1019 = llvm.add %1018, %3 : i64
    %1020 = llvm.call @_mlir_memref_to_llvm_alloc(%1019) : (i64) -> !llvm.ptr
    %1021 = llvm.ptrtoint %1020 : !llvm.ptr to i64
    %1022 = llvm.sub %3, %36 : i64
    %1023 = llvm.add %1021, %1022 : i64
    %1024 = llvm.urem %1023, %3 : i64
    %1025 = llvm.sub %1023, %1024 : i64
    %1026 = llvm.inttoptr %1025 : i64 to !llvm.ptr
    llvm.br ^bb164(%14 : i64)
  ^bb164(%1027: i64):  // 2 preds: ^bb163, ^bb165
    %1028 = llvm.icmp "slt" %1027, %37 : i64
    llvm.cond_br %1028, ^bb165, ^bb166
  ^bb165:  // pred: ^bb164
    %1029 = llvm.getelementptr inbounds %981[%1027] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %1030 = llvm.load %1029 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %1031 = llvm.trunc %1030 : i64 to i32
    %1032 = llvm.getelementptr inbounds %1026[%1027] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %1031, %1032 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %1033 = llvm.add %1027, %36 : i64
    llvm.br ^bb164(%1033 : i64)
  ^bb166:  // pred: ^bb164
    llvm.call @_mlir_memref_to_llvm_free(%975) : (!llvm.ptr) -> ()
    %1034 = llvm.getelementptr %11[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %1035 = llvm.ptrtoint %1034 : !llvm.ptr to i64
    %1036 = llvm.add %1035, %3 : i64
    %1037 = llvm.call @_mlir_memref_to_llvm_alloc(%1036) : (i64) -> !llvm.ptr
    %1038 = llvm.ptrtoint %1037 : !llvm.ptr to i64
    %1039 = llvm.sub %3, %36 : i64
    %1040 = llvm.add %1038, %1039 : i64
    %1041 = llvm.urem %1040, %3 : i64
    %1042 = llvm.sub %1040, %1041 : i64
    %1043 = llvm.inttoptr %1042 : i64 to !llvm.ptr
    llvm.br ^bb167(%14 : i64)
  ^bb167(%1044: i64):  // 2 preds: ^bb166, ^bb168
    %1045 = llvm.icmp "slt" %1044, %37 : i64
    llvm.cond_br %1045, ^bb168, ^bb169
  ^bb168:  // pred: ^bb167
    %1046 = llvm.getelementptr inbounds %952[%1044] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %1047 = llvm.load %1046 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %1048 = llvm.trunc %1047 : i64 to i32
    %1049 = llvm.getelementptr inbounds %1043[%1044] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %1048, %1049 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %1050 = llvm.add %1044, %36 : i64
    llvm.br ^bb167(%1050 : i64)
  ^bb169:  // pred: ^bb167
    llvm.call @_mlir_memref_to_llvm_free(%946) : (!llvm.ptr) -> ()
    %1051 = llvm.xor %940, %942 : i32
    %1052 = llvm.xor %1051, %30 : i32
    llvm.br ^bb170(%14 : i64)
  ^bb170(%1053: i64):  // 2 preds: ^bb169, ^bb171
    %1054 = llvm.icmp "slt" %1053, %37 : i64
    llvm.cond_br %1054, ^bb171, ^bb172
  ^bb171:  // pred: ^bb170
    %1055 = llvm.getelementptr inbounds %913[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %1056 = llvm.load %1055 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1057 = llvm.getelementptr inbounds %1016[%1053] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %1056, %1057 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %1058 = llvm.add %1053, %36 : i64
    llvm.br ^bb170(%1058 : i64)
  ^bb172:  // pred: ^bb170
    %1059 = llvm.getelementptr %11[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %1060 = llvm.ptrtoint %1059 : !llvm.ptr to i64
    %1061 = llvm.add %1060, %3 : i64
    %1062 = llvm.call @_mlir_memref_to_llvm_alloc(%1061) : (i64) -> !llvm.ptr
    %1063 = llvm.ptrtoint %1062 : !llvm.ptr to i64
    %1064 = llvm.sub %3, %36 : i64
    %1065 = llvm.add %1063, %1064 : i64
    %1066 = llvm.urem %1065, %3 : i64
    %1067 = llvm.sub %1065, %1066 : i64
    %1068 = llvm.inttoptr %1067 : i64 to !llvm.ptr
    llvm.br ^bb173(%14 : i64)
  ^bb173(%1069: i64):  // 2 preds: ^bb172, ^bb174
    %1070 = llvm.icmp "slt" %1069, %37 : i64
    llvm.cond_br %1070, ^bb174, ^bb175
  ^bb174:  // pred: ^bb173
    %1071 = llvm.getelementptr inbounds %1043[%1069] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %1072 = llvm.load %1071 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1073 = llvm.getelementptr inbounds %1016[%1069] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %1074 = llvm.load %1073 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1075 = llvm.add %1072, %1074 : i32
    %1076 = llvm.getelementptr inbounds %1068[%1069] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %1075, %1076 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %1077 = llvm.add %1069, %36 : i64
    llvm.br ^bb173(%1077 : i64)
  ^bb175:  // pred: ^bb173
    llvm.call @_mlir_memref_to_llvm_free(%1037) : (!llvm.ptr) -> ()
    llvm.br ^bb176(%14 : i64)
  ^bb176(%1078: i64):  // 2 preds: ^bb175, ^bb177
    %1079 = llvm.icmp "slt" %1078, %37 : i64
    llvm.cond_br %1079, ^bb177, ^bb178
  ^bb177:  // pred: ^bb176
    %1080 = llvm.getelementptr inbounds %913[3] : (!llvm.ptr) -> !llvm.ptr, i32
    %1081 = llvm.load %1080 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1082 = llvm.getelementptr inbounds %1016[%1078] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %1081, %1082 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %1083 = llvm.add %1078, %36 : i64
    llvm.br ^bb176(%1083 : i64)
  ^bb178:  // pred: ^bb176
    llvm.br ^bb179(%14 : i64)
  ^bb179(%1084: i64):  // 2 preds: ^bb178, ^bb180
    %1085 = llvm.icmp "slt" %1084, %37 : i64
    llvm.cond_br %1085, ^bb180, ^bb181
  ^bb180:  // pred: ^bb179
    %1086 = llvm.getelementptr inbounds %1026[%1084] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %1087 = llvm.load %1086 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1088 = llvm.getelementptr inbounds %1016[%1084] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %1089 = llvm.load %1088 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1090 = llvm.add %1087, %1089 : i32
    %1091 = llvm.getelementptr inbounds %1016[%1084] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %1090, %1091 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %1092 = llvm.add %1084, %36 : i64
    llvm.br ^bb179(%1092 : i64)
  ^bb181:  // pred: ^bb179
    llvm.call @_mlir_memref_to_llvm_free(%1020) : (!llvm.ptr) -> ()
    %1093 = llvm.getelementptr inbounds %913[3] : (!llvm.ptr) -> !llvm.ptr, i32
    %1094 = llvm.load %1093 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1095 = llvm.getelementptr inbounds %913[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %1096 = llvm.load %1095 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1097 = llvm.getelementptr %11[4] : (!llvm.ptr) -> !llvm.ptr, i32
    %1098 = llvm.ptrtoint %1097 : !llvm.ptr to i64
    %1099 = llvm.add %1098, %3 : i64
    %1100 = llvm.call @_mlir_memref_to_llvm_alloc(%1099) : (i64) -> !llvm.ptr
    %1101 = llvm.ptrtoint %1100 : !llvm.ptr to i64
    %1102 = llvm.sub %3, %36 : i64
    %1103 = llvm.add %1101, %1102 : i64
    %1104 = llvm.urem %1103, %3 : i64
    %1105 = llvm.sub %1103, %1104 : i64
    %1106 = llvm.inttoptr %1105 : i64 to !llvm.ptr
    %1107 = llvm.mul %38, %36 : i64
    %1108 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %1109 = llvm.ptrtoint %1108 : !llvm.ptr to i64
    %1110 = llvm.mul %1107, %1109 : i64
    "llvm.intr.memcpy"(%1106, %47, %1110) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %1111 = llvm.getelementptr %11[4] : (!llvm.ptr) -> !llvm.ptr, i32
    %1112 = llvm.ptrtoint %1111 : !llvm.ptr to i64
    %1113 = llvm.add %1112, %3 : i64
    %1114 = llvm.call @_mlir_memref_to_llvm_alloc(%1113) : (i64) -> !llvm.ptr
    %1115 = llvm.ptrtoint %1114 : !llvm.ptr to i64
    %1116 = llvm.sub %3, %36 : i64
    %1117 = llvm.add %1115, %1116 : i64
    %1118 = llvm.urem %1117, %3 : i64
    %1119 = llvm.sub %1117, %1118 : i64
    %1120 = llvm.inttoptr %1119 : i64 to !llvm.ptr
    %1121 = llvm.mul %38, %36 : i64
    %1122 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %1123 = llvm.ptrtoint %1122 : !llvm.ptr to i64
    %1124 = llvm.mul %1121, %1123 : i64
    "llvm.intr.memcpy"(%1120, %46, %1124) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %1125 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %1126 = llvm.ptrtoint %1125 : !llvm.ptr to i64
    %1127 = llvm.add %1126, %3 : i64
    %1128 = llvm.call @_mlir_memref_to_llvm_alloc(%1127) : (i64) -> !llvm.ptr
    %1129 = llvm.ptrtoint %1128 : !llvm.ptr to i64
    %1130 = llvm.sub %3, %36 : i64
    %1131 = llvm.add %1129, %1130 : i64
    %1132 = llvm.urem %1131, %3 : i64
    %1133 = llvm.sub %1131, %1132 : i64
    %1134 = llvm.inttoptr %1133 : i64 to !llvm.ptr
    %1135 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %1136 = llvm.ptrtoint %1135 : !llvm.ptr to i64
    %1137 = llvm.add %1136, %3 : i64
    %1138 = llvm.call @_mlir_memref_to_llvm_alloc(%1137) : (i64) -> !llvm.ptr
    %1139 = llvm.ptrtoint %1138 : !llvm.ptr to i64
    %1140 = llvm.sub %3, %36 : i64
    %1141 = llvm.add %1139, %1140 : i64
    %1142 = llvm.urem %1141, %3 : i64
    %1143 = llvm.sub %1141, %1142 : i64
    %1144 = llvm.inttoptr %1143 : i64 to !llvm.ptr
    %1145 = llvm.getelementptr %11[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %1146 = llvm.ptrtoint %1145 : !llvm.ptr to i64
    %1147 = llvm.add %1146, %3 : i64
    %1148 = llvm.call @_mlir_memref_to_llvm_alloc(%1147) : (i64) -> !llvm.ptr
    %1149 = llvm.ptrtoint %1148 : !llvm.ptr to i64
    %1150 = llvm.sub %3, %36 : i64
    %1151 = llvm.add %1149, %1150 : i64
    %1152 = llvm.urem %1151, %3 : i64
    %1153 = llvm.sub %1151, %1152 : i64
    %1154 = llvm.inttoptr %1153 : i64 to !llvm.ptr
    %1155 = llvm.getelementptr %11[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %1156 = llvm.ptrtoint %1155 : !llvm.ptr to i64
    %1157 = llvm.add %1156, %3 : i64
    %1158 = llvm.call @_mlir_memref_to_llvm_alloc(%1157) : (i64) -> !llvm.ptr
    %1159 = llvm.ptrtoint %1158 : !llvm.ptr to i64
    %1160 = llvm.sub %3, %36 : i64
    %1161 = llvm.add %1159, %1160 : i64
    %1162 = llvm.urem %1161, %3 : i64
    %1163 = llvm.sub %1161, %1162 : i64
    %1164 = llvm.inttoptr %1163 : i64 to !llvm.ptr
    %1165 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %1166 = llvm.ptrtoint %1165 : !llvm.ptr to i64
    %1167 = llvm.add %1166, %3 : i64
    %1168 = llvm.call @_mlir_memref_to_llvm_alloc(%1167) : (i64) -> !llvm.ptr
    %1169 = llvm.ptrtoint %1168 : !llvm.ptr to i64
    %1170 = llvm.sub %3, %36 : i64
    %1171 = llvm.add %1169, %1170 : i64
    %1172 = llvm.urem %1171, %3 : i64
    %1173 = llvm.sub %1171, %1172 : i64
    %1174 = llvm.inttoptr %1173 : i64 to !llvm.ptr
    %1175 = llvm.getelementptr %11[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %1176 = llvm.ptrtoint %1175 : !llvm.ptr to i64
    %1177 = llvm.add %1176, %3 : i64
    %1178 = llvm.call @_mlir_memref_to_llvm_alloc(%1177) : (i64) -> !llvm.ptr
    %1179 = llvm.ptrtoint %1178 : !llvm.ptr to i64
    %1180 = llvm.sub %3, %36 : i64
    %1181 = llvm.add %1179, %1180 : i64
    %1182 = llvm.urem %1181, %3 : i64
    %1183 = llvm.sub %1181, %1182 : i64
    %1184 = llvm.inttoptr %1183 : i64 to !llvm.ptr
    %1185 = llvm.getelementptr %11[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %1186 = llvm.ptrtoint %1185 : !llvm.ptr to i64
    %1187 = llvm.add %1186, %3 : i64
    %1188 = llvm.call @_mlir_memref_to_llvm_alloc(%1187) : (i64) -> !llvm.ptr
    %1189 = llvm.ptrtoint %1188 : !llvm.ptr to i64
    %1190 = llvm.sub %3, %36 : i64
    %1191 = llvm.add %1189, %1190 : i64
    %1192 = llvm.urem %1191, %3 : i64
    %1193 = llvm.sub %1191, %1192 : i64
    %1194 = llvm.inttoptr %1193 : i64 to !llvm.ptr
    %1195 = llvm.getelementptr %11[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %1196 = llvm.ptrtoint %1195 : !llvm.ptr to i64
    %1197 = llvm.add %1196, %3 : i64
    %1198 = llvm.call @_mlir_memref_to_llvm_alloc(%1197) : (i64) -> !llvm.ptr
    %1199 = llvm.ptrtoint %1198 : !llvm.ptr to i64
    %1200 = llvm.sub %3, %36 : i64
    %1201 = llvm.add %1199, %1200 : i64
    %1202 = llvm.urem %1201, %3 : i64
    %1203 = llvm.sub %1201, %1202 : i64
    %1204 = llvm.inttoptr %1203 : i64 to !llvm.ptr
    %1205 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %1206 = llvm.ptrtoint %1205 : !llvm.ptr to i64
    %1207 = llvm.add %1206, %3 : i64
    %1208 = llvm.call @_mlir_memref_to_llvm_alloc(%1207) : (i64) -> !llvm.ptr
    %1209 = llvm.ptrtoint %1208 : !llvm.ptr to i64
    %1210 = llvm.sub %3, %36 : i64
    %1211 = llvm.add %1209, %1210 : i64
    %1212 = llvm.urem %1211, %3 : i64
    %1213 = llvm.sub %1211, %1212 : i64
    %1214 = llvm.inttoptr %1213 : i64 to !llvm.ptr
    %1215 = llvm.getelementptr %11[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %1216 = llvm.ptrtoint %1215 : !llvm.ptr to i64
    %1217 = llvm.add %1216, %3 : i64
    %1218 = llvm.call @_mlir_memref_to_llvm_alloc(%1217) : (i64) -> !llvm.ptr
    %1219 = llvm.ptrtoint %1218 : !llvm.ptr to i64
    %1220 = llvm.sub %3, %36 : i64
    %1221 = llvm.add %1219, %1220 : i64
    %1222 = llvm.urem %1221, %3 : i64
    %1223 = llvm.sub %1221, %1222 : i64
    %1224 = llvm.inttoptr %1223 : i64 to !llvm.ptr
    %1225 = llvm.getelementptr %11[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %1226 = llvm.ptrtoint %1225 : !llvm.ptr to i64
    %1227 = llvm.add %1226, %3 : i64
    %1228 = llvm.call @_mlir_memref_to_llvm_alloc(%1227) : (i64) -> !llvm.ptr
    %1229 = llvm.ptrtoint %1228 : !llvm.ptr to i64
    %1230 = llvm.sub %3, %36 : i64
    %1231 = llvm.add %1229, %1230 : i64
    %1232 = llvm.urem %1231, %3 : i64
    %1233 = llvm.sub %1231, %1232 : i64
    %1234 = llvm.inttoptr %1233 : i64 to !llvm.ptr
    %1235 = llvm.getelementptr %11[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %1236 = llvm.ptrtoint %1235 : !llvm.ptr to i64
    %1237 = llvm.add %1236, %3 : i64
    %1238 = llvm.call @_mlir_memref_to_llvm_alloc(%1237) : (i64) -> !llvm.ptr
    %1239 = llvm.ptrtoint %1238 : !llvm.ptr to i64
    %1240 = llvm.sub %3, %36 : i64
    %1241 = llvm.add %1239, %1240 : i64
    %1242 = llvm.urem %1241, %3 : i64
    %1243 = llvm.sub %1241, %1242 : i64
    %1244 = llvm.inttoptr %1243 : i64 to !llvm.ptr
    %1245 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %1246 = llvm.ptrtoint %1245 : !llvm.ptr to i64
    %1247 = llvm.add %1246, %3 : i64
    %1248 = llvm.call @_mlir_memref_to_llvm_alloc(%1247) : (i64) -> !llvm.ptr
    %1249 = llvm.ptrtoint %1248 : !llvm.ptr to i64
    %1250 = llvm.sub %3, %36 : i64
    %1251 = llvm.add %1249, %1250 : i64
    %1252 = llvm.urem %1251, %3 : i64
    %1253 = llvm.sub %1251, %1252 : i64
    %1254 = llvm.inttoptr %1253 : i64 to !llvm.ptr
    %1255 = llvm.getelementptr %11[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %1256 = llvm.ptrtoint %1255 : !llvm.ptr to i64
    %1257 = llvm.add %1256, %3 : i64
    %1258 = llvm.call @_mlir_memref_to_llvm_alloc(%1257) : (i64) -> !llvm.ptr
    %1259 = llvm.ptrtoint %1258 : !llvm.ptr to i64
    %1260 = llvm.sub %3, %36 : i64
    %1261 = llvm.add %1259, %1260 : i64
    %1262 = llvm.urem %1261, %3 : i64
    %1263 = llvm.sub %1261, %1262 : i64
    %1264 = llvm.inttoptr %1263 : i64 to !llvm.ptr
    %1265 = llvm.getelementptr %11[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %1266 = llvm.ptrtoint %1265 : !llvm.ptr to i64
    %1267 = llvm.add %1266, %3 : i64
    %1268 = llvm.call @_mlir_memref_to_llvm_alloc(%1267) : (i64) -> !llvm.ptr
    %1269 = llvm.ptrtoint %1268 : !llvm.ptr to i64
    %1270 = llvm.sub %3, %36 : i64
    %1271 = llvm.add %1269, %1270 : i64
    %1272 = llvm.urem %1271, %3 : i64
    %1273 = llvm.sub %1271, %1272 : i64
    %1274 = llvm.inttoptr %1273 : i64 to !llvm.ptr
    %1275 = llvm.getelementptr %11[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %1276 = llvm.ptrtoint %1275 : !llvm.ptr to i64
    %1277 = llvm.add %1276, %3 : i64
    %1278 = llvm.call @_mlir_memref_to_llvm_alloc(%1277) : (i64) -> !llvm.ptr
    %1279 = llvm.ptrtoint %1278 : !llvm.ptr to i64
    %1280 = llvm.sub %3, %36 : i64
    %1281 = llvm.add %1279, %1280 : i64
    %1282 = llvm.urem %1281, %3 : i64
    %1283 = llvm.sub %1281, %1282 : i64
    %1284 = llvm.inttoptr %1283 : i64 to !llvm.ptr
    %1285 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %1286 = llvm.ptrtoint %1285 : !llvm.ptr to i64
    %1287 = llvm.add %1286, %3 : i64
    %1288 = llvm.call @_mlir_memref_to_llvm_alloc(%1287) : (i64) -> !llvm.ptr
    %1289 = llvm.ptrtoint %1288 : !llvm.ptr to i64
    %1290 = llvm.sub %3, %36 : i64
    %1291 = llvm.add %1289, %1290 : i64
    %1292 = llvm.urem %1291, %3 : i64
    %1293 = llvm.sub %1291, %1292 : i64
    %1294 = llvm.inttoptr %1293 : i64 to !llvm.ptr
    %1295 = llvm.getelementptr %11[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %1296 = llvm.ptrtoint %1295 : !llvm.ptr to i64
    %1297 = llvm.add %1296, %3 : i64
    %1298 = llvm.call @_mlir_memref_to_llvm_alloc(%1297) : (i64) -> !llvm.ptr
    %1299 = llvm.ptrtoint %1298 : !llvm.ptr to i64
    %1300 = llvm.sub %3, %36 : i64
    %1301 = llvm.add %1299, %1300 : i64
    %1302 = llvm.urem %1301, %3 : i64
    %1303 = llvm.sub %1301, %1302 : i64
    %1304 = llvm.inttoptr %1303 : i64 to !llvm.ptr
    %1305 = llvm.getelementptr %11[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %1306 = llvm.ptrtoint %1305 : !llvm.ptr to i64
    %1307 = llvm.add %1306, %3 : i64
    %1308 = llvm.call @_mlir_memref_to_llvm_alloc(%1307) : (i64) -> !llvm.ptr
    %1309 = llvm.ptrtoint %1308 : !llvm.ptr to i64
    %1310 = llvm.sub %3, %36 : i64
    %1311 = llvm.add %1309, %1310 : i64
    %1312 = llvm.urem %1311, %3 : i64
    %1313 = llvm.sub %1311, %1312 : i64
    %1314 = llvm.inttoptr %1313 : i64 to !llvm.ptr
    %1315 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %1316 = llvm.ptrtoint %1315 : !llvm.ptr to i64
    %1317 = llvm.add %1316, %3 : i64
    %1318 = llvm.call @_mlir_memref_to_llvm_alloc(%1317) : (i64) -> !llvm.ptr
    %1319 = llvm.ptrtoint %1318 : !llvm.ptr to i64
    %1320 = llvm.sub %3, %36 : i64
    %1321 = llvm.add %1319, %1320 : i64
    %1322 = llvm.urem %1321, %3 : i64
    %1323 = llvm.sub %1321, %1322 : i64
    %1324 = llvm.inttoptr %1323 : i64 to !llvm.ptr
    %1325 = llvm.getelementptr %11[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %1326 = llvm.ptrtoint %1325 : !llvm.ptr to i64
    %1327 = llvm.call @_mlir_memref_to_llvm_alloc(%1326) : (i64) -> !llvm.ptr
    %1328 = llvm.insertvalue %1327, %42[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1329 = llvm.insertvalue %1327, %1328[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1330 = llvm.insertvalue %14, %1329[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1331 = llvm.insertvalue %37, %1330[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1332 = llvm.insertvalue %36, %1331[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1333 = llvm.mul %37, %36 : i64
    %1334 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %1335 = llvm.ptrtoint %1334 : !llvm.ptr to i64
    %1336 = llvm.mul %1333, %1335 : i64
    "llvm.intr.memcpy"(%1327, %1016, %1336) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%1010) : (!llvm.ptr) -> ()
    %1337 = llvm.getelementptr %11[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %1338 = llvm.ptrtoint %1337 : !llvm.ptr to i64
    %1339 = llvm.call @_mlir_memref_to_llvm_alloc(%1338) : (i64) -> !llvm.ptr
    %1340 = llvm.insertvalue %1339, %42[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1341 = llvm.insertvalue %1339, %1340[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1342 = llvm.insertvalue %14, %1341[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1343 = llvm.insertvalue %37, %1342[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1344 = llvm.insertvalue %36, %1343[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1345 = llvm.mul %37, %36 : i64
    %1346 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %1347 = llvm.ptrtoint %1346 : !llvm.ptr to i64
    %1348 = llvm.mul %1345, %1347 : i64
    "llvm.intr.memcpy"(%1339, %1068, %1348) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%1062) : (!llvm.ptr) -> ()
    %1349 = llvm.getelementptr %11[4] : (!llvm.ptr) -> !llvm.ptr, i32
    %1350 = llvm.ptrtoint %1349 : !llvm.ptr to i64
    %1351 = llvm.call @_mlir_memref_to_llvm_alloc(%1350) : (i64) -> !llvm.ptr
    %1352 = llvm.insertvalue %1351, %42[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1353 = llvm.insertvalue %1351, %1352[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1354 = llvm.insertvalue %14, %1353[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1355 = llvm.insertvalue %38, %1354[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1356 = llvm.insertvalue %36, %1355[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1357 = llvm.mul %38, %36 : i64
    %1358 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %1359 = llvm.ptrtoint %1358 : !llvm.ptr to i64
    %1360 = llvm.mul %1357, %1359 : i64
    "llvm.intr.memcpy"(%1351, %1106, %1360) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%1100) : (!llvm.ptr) -> ()
    %1361 = llvm.getelementptr %11[4] : (!llvm.ptr) -> !llvm.ptr, i32
    %1362 = llvm.ptrtoint %1361 : !llvm.ptr to i64
    %1363 = llvm.call @_mlir_memref_to_llvm_alloc(%1362) : (i64) -> !llvm.ptr
    %1364 = llvm.insertvalue %1363, %42[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1365 = llvm.insertvalue %1363, %1364[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1366 = llvm.insertvalue %14, %1365[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1367 = llvm.insertvalue %38, %1366[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1368 = llvm.insertvalue %36, %1367[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1369 = llvm.mul %38, %36 : i64
    %1370 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %1371 = llvm.ptrtoint %1370 : !llvm.ptr to i64
    %1372 = llvm.mul %1369, %1371 : i64
    "llvm.intr.memcpy"(%1363, %1120, %1372) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%1114) : (!llvm.ptr) -> ()
    llvm.br ^bb182(%18, %18, %1344, %1332, %1094, %1052, %1096, %1356, %1368 : i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, i32, i32, i32, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)
  ^bb182(%1373: i64, %1374: i64, %1375: !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, %1376: !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, %1377: i32, %1378: i32, %1379: i32, %1380: !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, %1381: !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>):  // 2 preds: ^bb181, ^bb285
    %1382 = llvm.icmp "slt" %1373, %27 : i64
    llvm.cond_br %1382, ^bb183, ^bb286
  ^bb183:  // pred: ^bb182
    llvm.store %1377, %1134 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    llvm.store %1378, %1144 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %1383 = llvm.add %1374, %28 : i64
    %1384 = llvm.extractvalue %1380[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1385 = llvm.load %1384 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1386 = llvm.getelementptr %11[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %1387 = llvm.ptrtoint %1386 : !llvm.ptr to i64
    %1388 = llvm.add %1387, %3 : i64
    %1389 = llvm.call @_mlir_memref_to_llvm_alloc(%1388) : (i64) -> !llvm.ptr
    %1390 = llvm.ptrtoint %1389 : !llvm.ptr to i64
    %1391 = llvm.sub %3, %36 : i64
    %1392 = llvm.add %1390, %1391 : i64
    %1393 = llvm.urem %1392, %3 : i64
    %1394 = llvm.sub %1392, %1393 : i64
    %1395 = llvm.inttoptr %1394 : i64 to !llvm.ptr
    llvm.br ^bb184(%14 : i64)
  ^bb184(%1396: i64):  // 2 preds: ^bb183, ^bb185
    %1397 = llvm.icmp "slt" %1396, %37 : i64
    llvm.cond_br %1397, ^bb185, ^bb186
  ^bb185:  // pred: ^bb184
    %1398 = llvm.extractvalue %1375[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1399 = llvm.getelementptr inbounds %1398[%1396] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %1400 = llvm.load %1399 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1401 = llvm.extractvalue %1376[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1402 = llvm.getelementptr inbounds %1401[%1396] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %1403 = llvm.load %1402 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1404 = llvm.add %1400, %1403 : i32
    %1405 = llvm.getelementptr inbounds %1154[%1396] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %1404, %1405 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %1406 = llvm.add %1396, %36 : i64
    llvm.br ^bb184(%1406 : i64)
  ^bb186:  // pred: ^bb184
    %1407 = llvm.extractvalue %1375[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.call @_mlir_memref_to_llvm_free(%1407) : (!llvm.ptr) -> ()
    llvm.br ^bb187(%14 : i64)
  ^bb187(%1408: i64):  // 2 preds: ^bb186, ^bb188
    %1409 = llvm.icmp "slt" %1408, %37 : i64
    llvm.cond_br %1409, ^bb188, ^bb189
  ^bb188:  // pred: ^bb187
    %1410 = llvm.load %1384 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1411 = llvm.getelementptr inbounds %1395[%1408] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %1410, %1411 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %1412 = llvm.add %1408, %36 : i64
    llvm.br ^bb187(%1412 : i64)
  ^bb189:  // pred: ^bb187
    llvm.br ^bb190(%14 : i64)
  ^bb190(%1413: i64):  // 2 preds: ^bb189, ^bb191
    %1414 = llvm.icmp "slt" %1413, %37 : i64
    llvm.cond_br %1414, ^bb191, ^bb192
  ^bb191:  // pred: ^bb190
    %1415 = llvm.extractvalue %1376[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1416 = llvm.getelementptr inbounds %1415[%1413] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %1417 = llvm.load %1416 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1418 = llvm.getelementptr inbounds %1395[%1413] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %1419 = llvm.load %1418 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1420 = llvm.shl %1417, %1419 : i32
    %1421 = llvm.icmp "ult" %1419, %26 : i32
    %1422 = llvm.select %1421, %1420, %25 : i1, i32
    %1423 = llvm.getelementptr inbounds %1164[%1413] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %1422, %1423 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %1424 = llvm.add %1413, %36 : i64
    llvm.br ^bb190(%1424 : i64)
  ^bb192:  // pred: ^bb190
    %1425 = llvm.sub %26, %1385 : i32
    llvm.store %1425, %1174 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    llvm.br ^bb193(%14 : i64)
  ^bb193(%1426: i64):  // 2 preds: ^bb192, ^bb194
    %1427 = llvm.icmp "slt" %1426, %37 : i64
    llvm.cond_br %1427, ^bb194, ^bb195
  ^bb194:  // pred: ^bb193
    %1428 = llvm.load %1174 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1429 = llvm.getelementptr inbounds %1395[%1426] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %1428, %1429 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %1430 = llvm.add %1426, %36 : i64
    llvm.br ^bb193(%1430 : i64)
  ^bb195:  // pred: ^bb193
    llvm.br ^bb196(%14 : i64)
  ^bb196(%1431: i64):  // 2 preds: ^bb195, ^bb197
    %1432 = llvm.icmp "slt" %1431, %37 : i64
    llvm.cond_br %1432, ^bb197, ^bb198
  ^bb197:  // pred: ^bb196
    %1433 = llvm.extractvalue %1376[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1434 = llvm.getelementptr inbounds %1433[%1431] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %1435 = llvm.load %1434 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1436 = llvm.getelementptr inbounds %1395[%1431] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %1437 = llvm.load %1436 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1438 = llvm.lshr %1435, %1437 : i32
    %1439 = llvm.icmp "ult" %1437, %26 : i32
    %1440 = llvm.select %1439, %1438, %25 : i1, i32
    %1441 = llvm.getelementptr inbounds %1395[%1431] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %1440, %1441 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %1442 = llvm.add %1431, %36 : i64
    llvm.br ^bb196(%1442 : i64)
  ^bb198:  // pred: ^bb196
    %1443 = llvm.extractvalue %1376[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.call @_mlir_memref_to_llvm_free(%1443) : (!llvm.ptr) -> ()
    llvm.br ^bb199(%14 : i64)
  ^bb199(%1444: i64):  // 2 preds: ^bb198, ^bb200
    %1445 = llvm.icmp "slt" %1444, %37 : i64
    llvm.cond_br %1445, ^bb200, ^bb201
  ^bb200:  // pred: ^bb199
    %1446 = llvm.getelementptr inbounds %1164[%1444] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %1447 = llvm.load %1446 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1448 = llvm.getelementptr inbounds %1395[%1444] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %1449 = llvm.load %1448 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1450 = llvm.or %1447, %1449 : i32
    %1451 = llvm.getelementptr inbounds %1395[%1444] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %1450, %1451 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %1452 = llvm.add %1444, %36 : i64
    llvm.br ^bb199(%1452 : i64)
  ^bb201:  // pred: ^bb199
    llvm.br ^bb202(%14 : i64)
  ^bb202(%1453: i64):  // 2 preds: ^bb201, ^bb203
    %1454 = llvm.icmp "slt" %1453, %37 : i64
    llvm.cond_br %1454, ^bb203, ^bb204
  ^bb203:  // pred: ^bb202
    %1455 = llvm.getelementptr inbounds %1154[%1453] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %1456 = llvm.load %1455 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1457 = llvm.getelementptr inbounds %1395[%1453] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %1458 = llvm.load %1457 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1459 = llvm.xor %1456, %1458 : i32
    %1460 = llvm.getelementptr inbounds %1184[%1453] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %1459, %1460 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %1461 = llvm.add %1453, %36 : i64
    llvm.br ^bb202(%1461 : i64)
  ^bb204:  // pred: ^bb202
    %1462 = llvm.extractvalue %1380[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1463 = llvm.getelementptr inbounds %1462[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %1464 = llvm.load %1463 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    llvm.br ^bb205(%14 : i64)
  ^bb205(%1465: i64):  // 2 preds: ^bb204, ^bb206
    %1466 = llvm.icmp "slt" %1465, %37 : i64
    llvm.cond_br %1466, ^bb206, ^bb207
  ^bb206:  // pred: ^bb205
    %1467 = llvm.getelementptr inbounds %1154[%1465] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %1468 = llvm.load %1467 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1469 = llvm.getelementptr inbounds %1184[%1465] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %1470 = llvm.load %1469 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1471 = llvm.add %1468, %1470 : i32
    %1472 = llvm.getelementptr inbounds %1194[%1465] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %1471, %1472 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %1473 = llvm.add %1465, %36 : i64
    llvm.br ^bb205(%1473 : i64)
  ^bb207:  // pred: ^bb205
    llvm.br ^bb208(%14 : i64)
  ^bb208(%1474: i64):  // 2 preds: ^bb207, ^bb209
    %1475 = llvm.icmp "slt" %1474, %37 : i64
    llvm.cond_br %1475, ^bb209, ^bb210
  ^bb209:  // pred: ^bb208
    %1476 = llvm.getelementptr inbounds %1462[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %1477 = llvm.load %1476 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1478 = llvm.getelementptr inbounds %1395[%1474] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %1477, %1478 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %1479 = llvm.add %1474, %36 : i64
    llvm.br ^bb208(%1479 : i64)
  ^bb210:  // pred: ^bb208
    llvm.br ^bb211(%14 : i64)
  ^bb211(%1480: i64):  // 2 preds: ^bb210, ^bb212
    %1481 = llvm.icmp "slt" %1480, %37 : i64
    llvm.cond_br %1481, ^bb212, ^bb213
  ^bb212:  // pred: ^bb211
    %1482 = llvm.getelementptr inbounds %1184[%1480] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %1483 = llvm.load %1482 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1484 = llvm.getelementptr inbounds %1395[%1480] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %1485 = llvm.load %1484 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1486 = llvm.shl %1483, %1485 : i32
    %1487 = llvm.icmp "ult" %1485, %26 : i32
    %1488 = llvm.select %1487, %1486, %25 : i1, i32
    %1489 = llvm.getelementptr inbounds %1204[%1480] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %1488, %1489 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %1490 = llvm.add %1480, %36 : i64
    llvm.br ^bb211(%1490 : i64)
  ^bb213:  // pred: ^bb211
    %1491 = llvm.sub %26, %1464 : i32
    llvm.store %1491, %1214 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    llvm.br ^bb214(%14 : i64)
  ^bb214(%1492: i64):  // 2 preds: ^bb213, ^bb215
    %1493 = llvm.icmp "slt" %1492, %37 : i64
    llvm.cond_br %1493, ^bb215, ^bb216
  ^bb215:  // pred: ^bb214
    %1494 = llvm.load %1214 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1495 = llvm.getelementptr inbounds %1395[%1492] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %1494, %1495 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %1496 = llvm.add %1492, %36 : i64
    llvm.br ^bb214(%1496 : i64)
  ^bb216:  // pred: ^bb214
    llvm.br ^bb217(%14 : i64)
  ^bb217(%1497: i64):  // 2 preds: ^bb216, ^bb218
    %1498 = llvm.icmp "slt" %1497, %37 : i64
    llvm.cond_br %1498, ^bb218, ^bb219
  ^bb218:  // pred: ^bb217
    %1499 = llvm.getelementptr inbounds %1184[%1497] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %1500 = llvm.load %1499 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1501 = llvm.getelementptr inbounds %1395[%1497] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %1502 = llvm.load %1501 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1503 = llvm.lshr %1500, %1502 : i32
    %1504 = llvm.icmp "ult" %1502, %26 : i32
    %1505 = llvm.select %1504, %1503, %25 : i1, i32
    %1506 = llvm.getelementptr inbounds %1395[%1497] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %1505, %1506 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %1507 = llvm.add %1497, %36 : i64
    llvm.br ^bb217(%1507 : i64)
  ^bb219:  // pred: ^bb217
    llvm.br ^bb220(%14 : i64)
  ^bb220(%1508: i64):  // 2 preds: ^bb219, ^bb221
    %1509 = llvm.icmp "slt" %1508, %37 : i64
    llvm.cond_br %1509, ^bb221, ^bb222
  ^bb221:  // pred: ^bb220
    %1510 = llvm.getelementptr inbounds %1204[%1508] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %1511 = llvm.load %1510 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1512 = llvm.getelementptr inbounds %1395[%1508] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %1513 = llvm.load %1512 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1514 = llvm.or %1511, %1513 : i32
    %1515 = llvm.getelementptr inbounds %1395[%1508] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %1514, %1515 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %1516 = llvm.add %1508, %36 : i64
    llvm.br ^bb220(%1516 : i64)
  ^bb222:  // pred: ^bb220
    llvm.br ^bb223(%14 : i64)
  ^bb223(%1517: i64):  // 2 preds: ^bb222, ^bb224
    %1518 = llvm.icmp "slt" %1517, %37 : i64
    llvm.cond_br %1518, ^bb224, ^bb225
  ^bb224:  // pred: ^bb223
    %1519 = llvm.getelementptr inbounds %1194[%1517] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %1520 = llvm.load %1519 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1521 = llvm.getelementptr inbounds %1395[%1517] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %1522 = llvm.load %1521 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1523 = llvm.xor %1520, %1522 : i32
    %1524 = llvm.getelementptr inbounds %1224[%1517] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %1523, %1524 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %1525 = llvm.add %1517, %36 : i64
    llvm.br ^bb223(%1525 : i64)
  ^bb225:  // pred: ^bb223
    %1526 = llvm.extractvalue %1380[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1527 = llvm.getelementptr inbounds %1526[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %1528 = llvm.load %1527 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    llvm.br ^bb226(%14 : i64)
  ^bb226(%1529: i64):  // 2 preds: ^bb225, ^bb227
    %1530 = llvm.icmp "slt" %1529, %37 : i64
    llvm.cond_br %1530, ^bb227, ^bb228
  ^bb227:  // pred: ^bb226
    %1531 = llvm.getelementptr inbounds %1194[%1529] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %1532 = llvm.load %1531 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1533 = llvm.getelementptr inbounds %1224[%1529] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %1534 = llvm.load %1533 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1535 = llvm.add %1532, %1534 : i32
    %1536 = llvm.getelementptr inbounds %1234[%1529] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %1535, %1536 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %1537 = llvm.add %1529, %36 : i64
    llvm.br ^bb226(%1537 : i64)
  ^bb228:  // pred: ^bb226
    llvm.br ^bb229(%14 : i64)
  ^bb229(%1538: i64):  // 2 preds: ^bb228, ^bb230
    %1539 = llvm.icmp "slt" %1538, %37 : i64
    llvm.cond_br %1539, ^bb230, ^bb231
  ^bb230:  // pred: ^bb229
    %1540 = llvm.getelementptr inbounds %1526[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %1541 = llvm.load %1540 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1542 = llvm.getelementptr inbounds %1395[%1538] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %1541, %1542 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %1543 = llvm.add %1538, %36 : i64
    llvm.br ^bb229(%1543 : i64)
  ^bb231:  // pred: ^bb229
    llvm.br ^bb232(%14 : i64)
  ^bb232(%1544: i64):  // 2 preds: ^bb231, ^bb233
    %1545 = llvm.icmp "slt" %1544, %37 : i64
    llvm.cond_br %1545, ^bb233, ^bb234
  ^bb233:  // pred: ^bb232
    %1546 = llvm.getelementptr inbounds %1224[%1544] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %1547 = llvm.load %1546 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1548 = llvm.getelementptr inbounds %1395[%1544] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %1549 = llvm.load %1548 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1550 = llvm.shl %1547, %1549 : i32
    %1551 = llvm.icmp "ult" %1549, %26 : i32
    %1552 = llvm.select %1551, %1550, %25 : i1, i32
    %1553 = llvm.getelementptr inbounds %1244[%1544] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %1552, %1553 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %1554 = llvm.add %1544, %36 : i64
    llvm.br ^bb232(%1554 : i64)
  ^bb234:  // pred: ^bb232
    %1555 = llvm.sub %26, %1528 : i32
    llvm.store %1555, %1254 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    llvm.br ^bb235(%14 : i64)
  ^bb235(%1556: i64):  // 2 preds: ^bb234, ^bb236
    %1557 = llvm.icmp "slt" %1556, %37 : i64
    llvm.cond_br %1557, ^bb236, ^bb237
  ^bb236:  // pred: ^bb235
    %1558 = llvm.load %1254 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1559 = llvm.getelementptr inbounds %1395[%1556] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %1558, %1559 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %1560 = llvm.add %1556, %36 : i64
    llvm.br ^bb235(%1560 : i64)
  ^bb237:  // pred: ^bb235
    llvm.br ^bb238(%14 : i64)
  ^bb238(%1561: i64):  // 2 preds: ^bb237, ^bb239
    %1562 = llvm.icmp "slt" %1561, %37 : i64
    llvm.cond_br %1562, ^bb239, ^bb240
  ^bb239:  // pred: ^bb238
    %1563 = llvm.getelementptr inbounds %1224[%1561] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %1564 = llvm.load %1563 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1565 = llvm.getelementptr inbounds %1395[%1561] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %1566 = llvm.load %1565 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1567 = llvm.lshr %1564, %1566 : i32
    %1568 = llvm.icmp "ult" %1566, %26 : i32
    %1569 = llvm.select %1568, %1567, %25 : i1, i32
    %1570 = llvm.getelementptr inbounds %1395[%1561] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %1569, %1570 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %1571 = llvm.add %1561, %36 : i64
    llvm.br ^bb238(%1571 : i64)
  ^bb240:  // pred: ^bb238
    llvm.br ^bb241(%14 : i64)
  ^bb241(%1572: i64):  // 2 preds: ^bb240, ^bb242
    %1573 = llvm.icmp "slt" %1572, %37 : i64
    llvm.cond_br %1573, ^bb242, ^bb243
  ^bb242:  // pred: ^bb241
    %1574 = llvm.getelementptr inbounds %1244[%1572] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %1575 = llvm.load %1574 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1576 = llvm.getelementptr inbounds %1395[%1572] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %1577 = llvm.load %1576 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1578 = llvm.or %1575, %1577 : i32
    %1579 = llvm.getelementptr inbounds %1395[%1572] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %1578, %1579 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %1580 = llvm.add %1572, %36 : i64
    llvm.br ^bb241(%1580 : i64)
  ^bb243:  // pred: ^bb241
    llvm.br ^bb244(%14 : i64)
  ^bb244(%1581: i64):  // 2 preds: ^bb243, ^bb245
    %1582 = llvm.icmp "slt" %1581, %37 : i64
    llvm.cond_br %1582, ^bb245, ^bb246
  ^bb245:  // pred: ^bb244
    %1583 = llvm.getelementptr inbounds %1234[%1581] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %1584 = llvm.load %1583 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1585 = llvm.getelementptr inbounds %1395[%1581] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %1586 = llvm.load %1585 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1587 = llvm.xor %1584, %1586 : i32
    %1588 = llvm.getelementptr inbounds %1264[%1581] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %1587, %1588 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %1589 = llvm.add %1581, %36 : i64
    llvm.br ^bb244(%1589 : i64)
  ^bb246:  // pred: ^bb244
    %1590 = llvm.extractvalue %1380[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1591 = llvm.getelementptr inbounds %1590[3] : (!llvm.ptr) -> !llvm.ptr, i32
    %1592 = llvm.load %1591 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    llvm.br ^bb247(%14 : i64)
  ^bb247(%1593: i64):  // 2 preds: ^bb246, ^bb248
    %1594 = llvm.icmp "slt" %1593, %37 : i64
    llvm.cond_br %1594, ^bb248, ^bb249
  ^bb248:  // pred: ^bb247
    %1595 = llvm.getelementptr inbounds %1234[%1593] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %1596 = llvm.load %1595 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1597 = llvm.getelementptr inbounds %1264[%1593] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %1598 = llvm.load %1597 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1599 = llvm.add %1596, %1598 : i32
    %1600 = llvm.getelementptr inbounds %1274[%1593] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %1599, %1600 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %1601 = llvm.add %1593, %36 : i64
    llvm.br ^bb247(%1601 : i64)
  ^bb249:  // pred: ^bb247
    llvm.br ^bb250(%14 : i64)
  ^bb250(%1602: i64):  // 2 preds: ^bb249, ^bb251
    %1603 = llvm.icmp "slt" %1602, %37 : i64
    llvm.cond_br %1603, ^bb251, ^bb252
  ^bb251:  // pred: ^bb250
    %1604 = llvm.getelementptr inbounds %1590[3] : (!llvm.ptr) -> !llvm.ptr, i32
    %1605 = llvm.load %1604 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1606 = llvm.getelementptr inbounds %1395[%1602] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %1605, %1606 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %1607 = llvm.add %1602, %36 : i64
    llvm.br ^bb250(%1607 : i64)
  ^bb252:  // pred: ^bb250
    llvm.br ^bb253(%14 : i64)
  ^bb253(%1608: i64):  // 2 preds: ^bb252, ^bb254
    %1609 = llvm.icmp "slt" %1608, %37 : i64
    llvm.cond_br %1609, ^bb254, ^bb255
  ^bb254:  // pred: ^bb253
    %1610 = llvm.getelementptr inbounds %1264[%1608] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %1611 = llvm.load %1610 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1612 = llvm.getelementptr inbounds %1395[%1608] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %1613 = llvm.load %1612 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1614 = llvm.shl %1611, %1613 : i32
    %1615 = llvm.icmp "ult" %1613, %26 : i32
    %1616 = llvm.select %1615, %1614, %25 : i1, i32
    %1617 = llvm.getelementptr inbounds %1284[%1608] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %1616, %1617 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %1618 = llvm.add %1608, %36 : i64
    llvm.br ^bb253(%1618 : i64)
  ^bb255:  // pred: ^bb253
    %1619 = llvm.sub %26, %1592 : i32
    llvm.store %1619, %1294 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    llvm.br ^bb256(%14 : i64)
  ^bb256(%1620: i64):  // 2 preds: ^bb255, ^bb257
    %1621 = llvm.icmp "slt" %1620, %37 : i64
    llvm.cond_br %1621, ^bb257, ^bb258
  ^bb257:  // pred: ^bb256
    %1622 = llvm.load %1294 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1623 = llvm.getelementptr inbounds %1395[%1620] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %1622, %1623 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %1624 = llvm.add %1620, %36 : i64
    llvm.br ^bb256(%1624 : i64)
  ^bb258:  // pred: ^bb256
    llvm.br ^bb259(%14 : i64)
  ^bb259(%1625: i64):  // 2 preds: ^bb258, ^bb260
    %1626 = llvm.icmp "slt" %1625, %37 : i64
    llvm.cond_br %1626, ^bb260, ^bb261
  ^bb260:  // pred: ^bb259
    %1627 = llvm.getelementptr inbounds %1264[%1625] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %1628 = llvm.load %1627 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1629 = llvm.getelementptr inbounds %1395[%1625] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %1630 = llvm.load %1629 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1631 = llvm.lshr %1628, %1630 : i32
    %1632 = llvm.icmp "ult" %1630, %26 : i32
    %1633 = llvm.select %1632, %1631, %25 : i1, i32
    %1634 = llvm.getelementptr inbounds %1395[%1625] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %1633, %1634 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %1635 = llvm.add %1625, %36 : i64
    llvm.br ^bb259(%1635 : i64)
  ^bb261:  // pred: ^bb259
    llvm.br ^bb262(%14 : i64)
  ^bb262(%1636: i64):  // 2 preds: ^bb261, ^bb263
    %1637 = llvm.icmp "slt" %1636, %37 : i64
    llvm.cond_br %1637, ^bb263, ^bb264
  ^bb263:  // pred: ^bb262
    %1638 = llvm.getelementptr inbounds %1284[%1636] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %1639 = llvm.load %1638 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1640 = llvm.getelementptr inbounds %1395[%1636] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %1641 = llvm.load %1640 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1642 = llvm.or %1639, %1641 : i32
    %1643 = llvm.getelementptr inbounds %1395[%1636] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %1642, %1643 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %1644 = llvm.add %1636, %36 : i64
    llvm.br ^bb262(%1644 : i64)
  ^bb264:  // pred: ^bb262
    llvm.br ^bb265(%14 : i64)
  ^bb265(%1645: i64):  // 2 preds: ^bb264, ^bb266
    %1646 = llvm.icmp "slt" %1645, %37 : i64
    llvm.cond_br %1646, ^bb266, ^bb267
  ^bb266:  // pred: ^bb265
    %1647 = llvm.getelementptr inbounds %1274[%1645] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %1648 = llvm.load %1647 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1649 = llvm.getelementptr inbounds %1395[%1645] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %1650 = llvm.load %1649 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1651 = llvm.xor %1648, %1650 : i32
    %1652 = llvm.getelementptr inbounds %1304[%1645] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %1651, %1652 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %1653 = llvm.add %1645, %36 : i64
    llvm.br ^bb265(%1653 : i64)
  ^bb267:  // pred: ^bb265
    llvm.br ^bb268(%14 : i64)
  ^bb268(%1654: i64):  // 2 preds: ^bb267, ^bb269
    %1655 = llvm.icmp "slt" %1654, %37 : i64
    llvm.cond_br %1655, ^bb269, ^bb270
  ^bb269:  // pred: ^bb268
    %1656 = llvm.load %1134 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1657 = llvm.getelementptr inbounds %1395[%1654] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %1656, %1657 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %1658 = llvm.add %1654, %36 : i64
    llvm.br ^bb268(%1658 : i64)
  ^bb270:  // pred: ^bb268
    %1659 = llvm.getelementptr %11[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %1660 = llvm.ptrtoint %1659 : !llvm.ptr to i64
    %1661 = llvm.add %1660, %3 : i64
    %1662 = llvm.call @_mlir_memref_to_llvm_alloc(%1661) : (i64) -> !llvm.ptr
    %1663 = llvm.ptrtoint %1662 : !llvm.ptr to i64
    %1664 = llvm.sub %3, %36 : i64
    %1665 = llvm.add %1663, %1664 : i64
    %1666 = llvm.urem %1665, %3 : i64
    %1667 = llvm.sub %1665, %1666 : i64
    %1668 = llvm.inttoptr %1667 : i64 to !llvm.ptr
    llvm.br ^bb271(%14 : i64)
  ^bb271(%1669: i64):  // 2 preds: ^bb270, ^bb272
    %1670 = llvm.icmp "slt" %1669, %37 : i64
    llvm.cond_br %1670, ^bb272, ^bb273
  ^bb272:  // pred: ^bb271
    %1671 = llvm.getelementptr inbounds %1274[%1669] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %1672 = llvm.load %1671 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1673 = llvm.getelementptr inbounds %1395[%1669] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %1674 = llvm.load %1673 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1675 = llvm.add %1672, %1674 : i32
    %1676 = llvm.getelementptr inbounds %1668[%1669] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %1675, %1676 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %1677 = llvm.add %1669, %36 : i64
    llvm.br ^bb271(%1677 : i64)
  ^bb273:  // pred: ^bb271
    llvm.br ^bb274(%14 : i64)
  ^bb274(%1678: i64):  // 2 preds: ^bb273, ^bb275
    %1679 = llvm.icmp "slt" %1678, %37 : i64
    llvm.cond_br %1679, ^bb275, ^bb276
  ^bb275:  // pred: ^bb274
    %1680 = llvm.load %1144 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1681 = llvm.getelementptr inbounds %1395[%1678] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %1680, %1681 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %1682 = llvm.add %1678, %36 : i64
    llvm.br ^bb274(%1682 : i64)
  ^bb276:  // pred: ^bb274
    llvm.br ^bb277(%14 : i64)
  ^bb277(%1683: i64):  // 2 preds: ^bb276, ^bb278
    %1684 = llvm.icmp "slt" %1683, %37 : i64
    llvm.cond_br %1684, ^bb278, ^bb279
  ^bb278:  // pred: ^bb277
    %1685 = llvm.getelementptr inbounds %1304[%1683] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %1686 = llvm.load %1685 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1687 = llvm.getelementptr inbounds %1395[%1683] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %1688 = llvm.load %1687 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1689 = llvm.add %1686, %1688 : i32
    %1690 = llvm.getelementptr inbounds %1314[%1683] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %1689, %1690 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %1691 = llvm.add %1683, %36 : i64
    llvm.br ^bb277(%1691 : i64)
  ^bb279:  // pred: ^bb277
    %1692 = llvm.trunc %1383 : i64 to i32
    llvm.store %1692, %1324 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    llvm.br ^bb280(%14 : i64)
  ^bb280(%1693: i64):  // 2 preds: ^bb279, ^bb281
    %1694 = llvm.icmp "slt" %1693, %37 : i64
    llvm.cond_br %1694, ^bb281, ^bb282
  ^bb281:  // pred: ^bb280
    %1695 = llvm.load %1324 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1696 = llvm.getelementptr inbounds %1395[%1693] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %1695, %1696 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %1697 = llvm.add %1693, %36 : i64
    llvm.br ^bb280(%1697 : i64)
  ^bb282:  // pred: ^bb280
    llvm.br ^bb283(%14 : i64)
  ^bb283(%1698: i64):  // 2 preds: ^bb282, ^bb284
    %1699 = llvm.icmp "slt" %1698, %37 : i64
    llvm.cond_br %1699, ^bb284, ^bb285
  ^bb284:  // pred: ^bb283
    %1700 = llvm.getelementptr inbounds %1314[%1698] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %1701 = llvm.load %1700 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1702 = llvm.getelementptr inbounds %1395[%1698] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %1703 = llvm.load %1702 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1704 = llvm.add %1701, %1703 : i32
    %1705 = llvm.getelementptr inbounds %1395[%1698] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %1704, %1705 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %1706 = llvm.add %1698, %36 : i64
    llvm.br ^bb283(%1706 : i64)
  ^bb285:  // pred: ^bb283
    %1707 = llvm.getelementptr %11[4] : (!llvm.ptr) -> !llvm.ptr, i32
    %1708 = llvm.ptrtoint %1707 : !llvm.ptr to i64
    %1709 = llvm.add %1708, %3 : i64
    %1710 = llvm.call @_mlir_memref_to_llvm_alloc(%1709) : (i64) -> !llvm.ptr
    %1711 = llvm.ptrtoint %1710 : !llvm.ptr to i64
    %1712 = llvm.sub %3, %36 : i64
    %1713 = llvm.add %1711, %1712 : i64
    %1714 = llvm.urem %1713, %3 : i64
    %1715 = llvm.sub %1713, %1714 : i64
    %1716 = llvm.inttoptr %1715 : i64 to !llvm.ptr
    %1717 = llvm.extractvalue %1381[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1718 = llvm.mul %1717, %36 : i64
    %1719 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %1720 = llvm.ptrtoint %1719 : !llvm.ptr to i64
    %1721 = llvm.mul %1718, %1720 : i64
    %1722 = llvm.extractvalue %1381[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1723 = llvm.extractvalue %1381[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1724 = llvm.getelementptr inbounds %1722[%1723] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    "llvm.intr.memcpy"(%1716, %1724, %1721) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %1725 = llvm.extractvalue %1381[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.call @_mlir_memref_to_llvm_free(%1725) : (!llvm.ptr) -> ()
    %1726 = llvm.getelementptr %11[4] : (!llvm.ptr) -> !llvm.ptr, i32
    %1727 = llvm.ptrtoint %1726 : !llvm.ptr to i64
    %1728 = llvm.add %1727, %3 : i64
    %1729 = llvm.call @_mlir_memref_to_llvm_alloc(%1728) : (i64) -> !llvm.ptr
    %1730 = llvm.ptrtoint %1729 : !llvm.ptr to i64
    %1731 = llvm.sub %3, %36 : i64
    %1732 = llvm.add %1730, %1731 : i64
    %1733 = llvm.urem %1732, %3 : i64
    %1734 = llvm.sub %1732, %1733 : i64
    %1735 = llvm.inttoptr %1734 : i64 to !llvm.ptr
    %1736 = llvm.extractvalue %1380[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1737 = llvm.mul %1736, %36 : i64
    %1738 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %1739 = llvm.ptrtoint %1738 : !llvm.ptr to i64
    %1740 = llvm.mul %1737, %1739 : i64
    %1741 = llvm.extractvalue %1380[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1742 = llvm.extractvalue %1380[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1743 = llvm.getelementptr inbounds %1741[%1742] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    "llvm.intr.memcpy"(%1735, %1743, %1740) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %1744 = llvm.extractvalue %1380[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.call @_mlir_memref_to_llvm_free(%1744) : (!llvm.ptr) -> ()
    %1745 = llvm.getelementptr %11[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %1746 = llvm.ptrtoint %1745 : !llvm.ptr to i64
    %1747 = llvm.call @_mlir_memref_to_llvm_alloc(%1746) : (i64) -> !llvm.ptr
    %1748 = llvm.insertvalue %1747, %42[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1749 = llvm.insertvalue %1747, %1748[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1750 = llvm.insertvalue %14, %1749[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1751 = llvm.insertvalue %37, %1750[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1752 = llvm.insertvalue %36, %1751[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1753 = llvm.mul %37, %36 : i64
    %1754 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %1755 = llvm.ptrtoint %1754 : !llvm.ptr to i64
    %1756 = llvm.mul %1753, %1755 : i64
    "llvm.intr.memcpy"(%1747, %1395, %1756) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%1389) : (!llvm.ptr) -> ()
    %1757 = llvm.getelementptr %11[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %1758 = llvm.ptrtoint %1757 : !llvm.ptr to i64
    %1759 = llvm.call @_mlir_memref_to_llvm_alloc(%1758) : (i64) -> !llvm.ptr
    %1760 = llvm.insertvalue %1759, %42[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1761 = llvm.insertvalue %1759, %1760[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1762 = llvm.insertvalue %14, %1761[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1763 = llvm.insertvalue %37, %1762[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1764 = llvm.insertvalue %36, %1763[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1765 = llvm.mul %37, %36 : i64
    %1766 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %1767 = llvm.ptrtoint %1766 : !llvm.ptr to i64
    %1768 = llvm.mul %1765, %1767 : i64
    "llvm.intr.memcpy"(%1759, %1668, %1768) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%1662) : (!llvm.ptr) -> ()
    %1769 = llvm.getelementptr %11[4] : (!llvm.ptr) -> !llvm.ptr, i32
    %1770 = llvm.ptrtoint %1769 : !llvm.ptr to i64
    %1771 = llvm.call @_mlir_memref_to_llvm_alloc(%1770) : (i64) -> !llvm.ptr
    %1772 = llvm.insertvalue %1771, %42[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1773 = llvm.insertvalue %1771, %1772[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1774 = llvm.insertvalue %14, %1773[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1775 = llvm.insertvalue %38, %1774[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1776 = llvm.insertvalue %36, %1775[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1777 = llvm.mul %38, %36 : i64
    %1778 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %1779 = llvm.ptrtoint %1778 : !llvm.ptr to i64
    %1780 = llvm.mul %1777, %1779 : i64
    "llvm.intr.memcpy"(%1771, %1716, %1780) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%1710) : (!llvm.ptr) -> ()
    %1781 = llvm.getelementptr %11[4] : (!llvm.ptr) -> !llvm.ptr, i32
    %1782 = llvm.ptrtoint %1781 : !llvm.ptr to i64
    %1783 = llvm.call @_mlir_memref_to_llvm_alloc(%1782) : (i64) -> !llvm.ptr
    %1784 = llvm.insertvalue %1783, %42[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1785 = llvm.insertvalue %1783, %1784[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1786 = llvm.insertvalue %14, %1785[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1787 = llvm.insertvalue %38, %1786[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1788 = llvm.insertvalue %36, %1787[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1789 = llvm.mul %38, %36 : i64
    %1790 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %1791 = llvm.ptrtoint %1790 : !llvm.ptr to i64
    %1792 = llvm.mul %1789, %1791 : i64
    "llvm.intr.memcpy"(%1783, %1735, %1792) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%1729) : (!llvm.ptr) -> ()
    %1793 = llvm.add %1373, %28 : i64
    llvm.br ^bb182(%1793, %1383, %1764, %1752, %1378, %1379, %1377, %1776, %1788 : i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, i32, i32, i32, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)
  ^bb286:  // pred: ^bb182
    %1794 = llvm.extractvalue %1381[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.call @_mlir_memref_to_llvm_free(%1794) : (!llvm.ptr) -> ()
    %1795 = llvm.extractvalue %1380[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.call @_mlir_memref_to_llvm_free(%1795) : (!llvm.ptr) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%1318) : (!llvm.ptr) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%1308) : (!llvm.ptr) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%1298) : (!llvm.ptr) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%1288) : (!llvm.ptr) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%1278) : (!llvm.ptr) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%1268) : (!llvm.ptr) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%1258) : (!llvm.ptr) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%1248) : (!llvm.ptr) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%1238) : (!llvm.ptr) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%1228) : (!llvm.ptr) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%1218) : (!llvm.ptr) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%1208) : (!llvm.ptr) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%1198) : (!llvm.ptr) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%1188) : (!llvm.ptr) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%1178) : (!llvm.ptr) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%1168) : (!llvm.ptr) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%1158) : (!llvm.ptr) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%1148) : (!llvm.ptr) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%1138) : (!llvm.ptr) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%1128) : (!llvm.ptr) -> ()
    %1796 = llvm.extractvalue %1375[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1797 = llvm.extractvalue %1376[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1798 = llvm.getelementptr %11[4] : (!llvm.ptr) -> !llvm.ptr, i32
    %1799 = llvm.ptrtoint %1798 : !llvm.ptr to i64
    %1800 = llvm.add %1799, %3 : i64
    %1801 = llvm.call @_mlir_memref_to_llvm_alloc(%1800) : (i64) -> !llvm.ptr
    %1802 = llvm.ptrtoint %1801 : !llvm.ptr to i64
    %1803 = llvm.sub %3, %36 : i64
    %1804 = llvm.add %1802, %1803 : i64
    %1805 = llvm.urem %1804, %3 : i64
    %1806 = llvm.sub %1804, %1805 : i64
    %1807 = llvm.inttoptr %1806 : i64 to !llvm.ptr
    llvm.br ^bb287(%14 : i64)
  ^bb287(%1808: i64):  // 2 preds: ^bb286, ^bb295
    %1809 = llvm.icmp "slt" %1808, %37 : i64
    llvm.cond_br %1809, ^bb288, ^bb296
  ^bb288:  // pred: ^bb287
    llvm.br ^bb289(%14 : i64)
  ^bb289(%1810: i64):  // 2 preds: ^bb288, ^bb294
    %1811 = llvm.icmp "slt" %1810, %37 : i64
    llvm.cond_br %1811, ^bb290, ^bb295
  ^bb290:  // pred: ^bb289
    %1812 = llvm.icmp "ult" %1810, %36 : i64
    llvm.cond_br %1812, ^bb291, ^bb292
  ^bb291:  // pred: ^bb290
    %1813 = llvm.add %1808, %1810 : i64
    %1814 = llvm.getelementptr inbounds %1796[%1813] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %1815 = llvm.load %1814 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    llvm.br ^bb293(%1815 : i32)
  ^bb292:  // pred: ^bb290
    %1816 = llvm.sub %1810, %36 : i64
    %1817 = llvm.add %1808, %1816 : i64
    %1818 = llvm.getelementptr inbounds %1797[%1817] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %1819 = llvm.load %1818 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    llvm.br ^bb293(%1819 : i32)
  ^bb293(%1820: i32):  // 2 preds: ^bb291, ^bb292
    llvm.br ^bb294
  ^bb294:  // pred: ^bb293
    %1821 = llvm.mul %1808, %37 : i64
    %1822 = llvm.add %1821, %1810 : i64
    %1823 = llvm.getelementptr inbounds %1807[%1822] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %1820, %1823 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %1824 = llvm.add %1810, %36 : i64
    llvm.br ^bb289(%1824 : i64)
  ^bb295:  // pred: ^bb289
    %1825 = llvm.add %1808, %36 : i64
    llvm.br ^bb287(%1825 : i64)
  ^bb296:  // pred: ^bb287
    %1826 = llvm.extractvalue %1375[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.call @_mlir_memref_to_llvm_free(%1826) : (!llvm.ptr) -> ()
    %1827 = llvm.extractvalue %1376[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.call @_mlir_memref_to_llvm_free(%1827) : (!llvm.ptr) -> ()
    %1828 = llvm.load %1807 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1829 = llvm.getelementptr inbounds %1807[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %1830 = llvm.load %1829 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1831 = llvm.xor %1828, %1830 : i32
    %1832 = llvm.xor %1831, %30 : i32
    %1833 = llvm.load %1807 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1834 = llvm.getelementptr inbounds %1807[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %1835 = llvm.load %1834 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1836 = llvm.getelementptr inbounds %1807[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %1837 = llvm.load %1836 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1838 = llvm.load %1807 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1839 = llvm.getelementptr %11[4] : (!llvm.ptr) -> !llvm.ptr, i32
    %1840 = llvm.ptrtoint %1839 : !llvm.ptr to i64
    %1841 = llvm.add %1840, %3 : i64
    %1842 = llvm.call @_mlir_memref_to_llvm_alloc(%1841) : (i64) -> !llvm.ptr
    %1843 = llvm.ptrtoint %1842 : !llvm.ptr to i64
    %1844 = llvm.sub %3, %36 : i64
    %1845 = llvm.add %1843, %1844 : i64
    %1846 = llvm.urem %1845, %3 : i64
    %1847 = llvm.sub %1845, %1846 : i64
    %1848 = llvm.inttoptr %1847 : i64 to !llvm.ptr
    %1849 = llvm.mul %38, %36 : i64
    %1850 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %1851 = llvm.ptrtoint %1850 : !llvm.ptr to i64
    %1852 = llvm.mul %1849, %1851 : i64
    "llvm.intr.memcpy"(%1848, %47, %1852) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %1853 = llvm.getelementptr %11[4] : (!llvm.ptr) -> !llvm.ptr, i32
    %1854 = llvm.ptrtoint %1853 : !llvm.ptr to i64
    %1855 = llvm.add %1854, %3 : i64
    %1856 = llvm.call @_mlir_memref_to_llvm_alloc(%1855) : (i64) -> !llvm.ptr
    %1857 = llvm.ptrtoint %1856 : !llvm.ptr to i64
    %1858 = llvm.sub %3, %36 : i64
    %1859 = llvm.add %1857, %1858 : i64
    %1860 = llvm.urem %1859, %3 : i64
    %1861 = llvm.sub %1859, %1860 : i64
    %1862 = llvm.inttoptr %1861 : i64 to !llvm.ptr
    %1863 = llvm.mul %38, %36 : i64
    %1864 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %1865 = llvm.ptrtoint %1864 : !llvm.ptr to i64
    %1866 = llvm.mul %1863, %1865 : i64
    "llvm.intr.memcpy"(%1862, %46, %1866) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %1867 = llvm.getelementptr %11[4] : (!llvm.ptr) -> !llvm.ptr, i32
    %1868 = llvm.ptrtoint %1867 : !llvm.ptr to i64
    %1869 = llvm.call @_mlir_memref_to_llvm_alloc(%1868) : (i64) -> !llvm.ptr
    %1870 = llvm.insertvalue %1869, %42[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1871 = llvm.insertvalue %1869, %1870[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1872 = llvm.insertvalue %14, %1871[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1873 = llvm.insertvalue %38, %1872[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1874 = llvm.insertvalue %36, %1873[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1875 = llvm.mul %38, %36 : i64
    %1876 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %1877 = llvm.ptrtoint %1876 : !llvm.ptr to i64
    %1878 = llvm.mul %1875, %1877 : i64
    "llvm.intr.memcpy"(%1869, %1848, %1878) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%1842) : (!llvm.ptr) -> ()
    %1879 = llvm.getelementptr %11[4] : (!llvm.ptr) -> !llvm.ptr, i32
    %1880 = llvm.ptrtoint %1879 : !llvm.ptr to i64
    %1881 = llvm.call @_mlir_memref_to_llvm_alloc(%1880) : (i64) -> !llvm.ptr
    %1882 = llvm.insertvalue %1881, %42[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1883 = llvm.insertvalue %1881, %1882[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1884 = llvm.insertvalue %14, %1883[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1885 = llvm.insertvalue %38, %1884[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1886 = llvm.insertvalue %36, %1885[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1887 = llvm.mul %38, %36 : i64
    %1888 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %1889 = llvm.ptrtoint %1888 : !llvm.ptr to i64
    %1890 = llvm.mul %1887, %1889 : i64
    "llvm.intr.memcpy"(%1881, %1862, %1890) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%1856) : (!llvm.ptr) -> ()
    llvm.br ^bb297(%18, %18, %1833, %1835, %1837, %1832, %1838, %1874, %1886 : i64, i64, i32, i32, i32, i32, i32, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)
  ^bb297(%1891: i64, %1892: i64, %1893: i32, %1894: i32, %1895: i32, %1896: i32, %1897: i32, %1898: !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, %1899: !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>):  // 2 preds: ^bb296, ^bb298
    %1900 = llvm.icmp "slt" %1891, %27 : i64
    llvm.cond_br %1900, ^bb298, ^bb299
  ^bb298:  // pred: ^bb297
    %1901 = llvm.add %1892, %28 : i64
    %1902 = llvm.extractvalue %1898[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1903 = llvm.load %1902 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1904 = llvm.add %1893, %1894 : i32
    %1905 = llvm.shl %1894, %1903 : i32
    %1906 = llvm.icmp "ult" %1903, %26 : i32
    %1907 = llvm.select %1906, %1905, %25 : i1, i32
    %1908 = llvm.sub %26, %1903 : i32
    %1909 = llvm.lshr %1894, %1908 : i32
    %1910 = llvm.icmp "ult" %1908, %26 : i32
    %1911 = llvm.select %1910, %1909, %25 : i1, i32
    %1912 = llvm.or %1907, %1911 : i32
    %1913 = llvm.xor %1904, %1912 : i32
    %1914 = llvm.extractvalue %1898[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1915 = llvm.getelementptr inbounds %1914[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %1916 = llvm.load %1915 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1917 = llvm.add %1904, %1913 : i32
    %1918 = llvm.shl %1913, %1916 : i32
    %1919 = llvm.icmp "ult" %1916, %26 : i32
    %1920 = llvm.select %1919, %1918, %25 : i1, i32
    %1921 = llvm.sub %26, %1916 : i32
    %1922 = llvm.lshr %1913, %1921 : i32
    %1923 = llvm.icmp "ult" %1921, %26 : i32
    %1924 = llvm.select %1923, %1922, %25 : i1, i32
    %1925 = llvm.or %1920, %1924 : i32
    %1926 = llvm.xor %1917, %1925 : i32
    %1927 = llvm.extractvalue %1898[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1928 = llvm.getelementptr inbounds %1927[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %1929 = llvm.load %1928 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1930 = llvm.add %1917, %1926 : i32
    %1931 = llvm.shl %1926, %1929 : i32
    %1932 = llvm.icmp "ult" %1929, %26 : i32
    %1933 = llvm.select %1932, %1931, %25 : i1, i32
    %1934 = llvm.sub %26, %1929 : i32
    %1935 = llvm.lshr %1926, %1934 : i32
    %1936 = llvm.icmp "ult" %1934, %26 : i32
    %1937 = llvm.select %1936, %1935, %25 : i1, i32
    %1938 = llvm.or %1933, %1937 : i32
    %1939 = llvm.xor %1930, %1938 : i32
    %1940 = llvm.extractvalue %1898[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1941 = llvm.getelementptr inbounds %1940[3] : (!llvm.ptr) -> !llvm.ptr, i32
    %1942 = llvm.load %1941 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %1943 = llvm.add %1930, %1939 : i32
    %1944 = llvm.shl %1939, %1942 : i32
    %1945 = llvm.icmp "ult" %1942, %26 : i32
    %1946 = llvm.select %1945, %1944, %25 : i1, i32
    %1947 = llvm.sub %26, %1942 : i32
    %1948 = llvm.lshr %1939, %1947 : i32
    %1949 = llvm.icmp "ult" %1947, %26 : i32
    %1950 = llvm.select %1949, %1948, %25 : i1, i32
    %1951 = llvm.or %1946, %1950 : i32
    %1952 = llvm.xor %1943, %1951 : i32
    %1953 = llvm.add %1943, %1895 : i32
    %1954 = llvm.add %1952, %1896 : i32
    %1955 = llvm.trunc %1901 : i64 to i32
    %1956 = llvm.add %1954, %1955 : i32
    %1957 = llvm.getelementptr %11[4] : (!llvm.ptr) -> !llvm.ptr, i32
    %1958 = llvm.ptrtoint %1957 : !llvm.ptr to i64
    %1959 = llvm.add %1958, %3 : i64
    %1960 = llvm.call @_mlir_memref_to_llvm_alloc(%1959) : (i64) -> !llvm.ptr
    %1961 = llvm.ptrtoint %1960 : !llvm.ptr to i64
    %1962 = llvm.sub %3, %36 : i64
    %1963 = llvm.add %1961, %1962 : i64
    %1964 = llvm.urem %1963, %3 : i64
    %1965 = llvm.sub %1963, %1964 : i64
    %1966 = llvm.inttoptr %1965 : i64 to !llvm.ptr
    %1967 = llvm.extractvalue %1899[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1968 = llvm.mul %1967, %36 : i64
    %1969 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %1970 = llvm.ptrtoint %1969 : !llvm.ptr to i64
    %1971 = llvm.mul %1968, %1970 : i64
    %1972 = llvm.extractvalue %1899[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1973 = llvm.extractvalue %1899[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1974 = llvm.getelementptr inbounds %1972[%1973] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    "llvm.intr.memcpy"(%1966, %1974, %1971) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %1975 = llvm.extractvalue %1899[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.call @_mlir_memref_to_llvm_free(%1975) : (!llvm.ptr) -> ()
    %1976 = llvm.getelementptr %11[4] : (!llvm.ptr) -> !llvm.ptr, i32
    %1977 = llvm.ptrtoint %1976 : !llvm.ptr to i64
    %1978 = llvm.add %1977, %3 : i64
    %1979 = llvm.call @_mlir_memref_to_llvm_alloc(%1978) : (i64) -> !llvm.ptr
    %1980 = llvm.ptrtoint %1979 : !llvm.ptr to i64
    %1981 = llvm.sub %3, %36 : i64
    %1982 = llvm.add %1980, %1981 : i64
    %1983 = llvm.urem %1982, %3 : i64
    %1984 = llvm.sub %1982, %1983 : i64
    %1985 = llvm.inttoptr %1984 : i64 to !llvm.ptr
    %1986 = llvm.extractvalue %1898[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1987 = llvm.mul %1986, %36 : i64
    %1988 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %1989 = llvm.ptrtoint %1988 : !llvm.ptr to i64
    %1990 = llvm.mul %1987, %1989 : i64
    %1991 = llvm.extractvalue %1898[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1992 = llvm.extractvalue %1898[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1993 = llvm.getelementptr inbounds %1991[%1992] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    "llvm.intr.memcpy"(%1985, %1993, %1990) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %1994 = llvm.extractvalue %1898[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.call @_mlir_memref_to_llvm_free(%1994) : (!llvm.ptr) -> ()
    %1995 = llvm.getelementptr %11[4] : (!llvm.ptr) -> !llvm.ptr, i32
    %1996 = llvm.ptrtoint %1995 : !llvm.ptr to i64
    %1997 = llvm.call @_mlir_memref_to_llvm_alloc(%1996) : (i64) -> !llvm.ptr
    %1998 = llvm.insertvalue %1997, %42[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %1999 = llvm.insertvalue %1997, %1998[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2000 = llvm.insertvalue %14, %1999[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2001 = llvm.insertvalue %38, %2000[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2002 = llvm.insertvalue %36, %2001[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2003 = llvm.mul %38, %36 : i64
    %2004 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %2005 = llvm.ptrtoint %2004 : !llvm.ptr to i64
    %2006 = llvm.mul %2003, %2005 : i64
    "llvm.intr.memcpy"(%1997, %1966, %2006) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%1960) : (!llvm.ptr) -> ()
    %2007 = llvm.getelementptr %11[4] : (!llvm.ptr) -> !llvm.ptr, i32
    %2008 = llvm.ptrtoint %2007 : !llvm.ptr to i64
    %2009 = llvm.call @_mlir_memref_to_llvm_alloc(%2008) : (i64) -> !llvm.ptr
    %2010 = llvm.insertvalue %2009, %42[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2011 = llvm.insertvalue %2009, %2010[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2012 = llvm.insertvalue %14, %2011[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2013 = llvm.insertvalue %38, %2012[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2014 = llvm.insertvalue %36, %2013[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2015 = llvm.mul %38, %36 : i64
    %2016 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %2017 = llvm.ptrtoint %2016 : !llvm.ptr to i64
    %2018 = llvm.mul %2015, %2017 : i64
    "llvm.intr.memcpy"(%2009, %1985, %2018) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%1979) : (!llvm.ptr) -> ()
    %2019 = llvm.add %1891, %28 : i64
    llvm.br ^bb297(%2019, %1901, %1953, %1956, %1896, %1897, %1895, %2002, %2014 : i64, i64, i32, i32, i32, i32, i32, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)
  ^bb299:  // pred: ^bb297
    %2020 = llvm.extractvalue %1899[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.call @_mlir_memref_to_llvm_free(%2020) : (!llvm.ptr) -> ()
    %2021 = llvm.extractvalue %1898[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.call @_mlir_memref_to_llvm_free(%2021) : (!llvm.ptr) -> ()
    %2022 = llvm.zext %1893 : i32 to i64
    %2023 = llvm.zext %1894 : i32 to i64
    %2024 = llvm.shl %2022, %17 : i64
    %2025 = llvm.or %2024, %2023 : i64
    %2026 = llvm.getelementptr inbounds %1807[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %2027 = llvm.load %2026 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %2028 = llvm.getelementptr inbounds %1807[3] : (!llvm.ptr) -> !llvm.ptr, i32
    %2029 = llvm.load %2028 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %2030 = llvm.xor %2027, %2029 : i32
    %2031 = llvm.xor %2030, %30 : i32
    %2032 = llvm.getelementptr inbounds %1807[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %2033 = llvm.load %2032 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %2034 = llvm.getelementptr inbounds %1807[3] : (!llvm.ptr) -> !llvm.ptr, i32
    %2035 = llvm.load %2034 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %2036 = llvm.getelementptr inbounds %1807[3] : (!llvm.ptr) -> !llvm.ptr, i32
    %2037 = llvm.load %2036 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %2038 = llvm.getelementptr inbounds %1807[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %2039 = llvm.load %2038 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    llvm.call @_mlir_memref_to_llvm_free(%1801) : (!llvm.ptr) -> ()
    %2040 = llvm.getelementptr %11[4] : (!llvm.ptr) -> !llvm.ptr, i32
    %2041 = llvm.ptrtoint %2040 : !llvm.ptr to i64
    %2042 = llvm.add %2041, %3 : i64
    %2043 = llvm.call @_mlir_memref_to_llvm_alloc(%2042) : (i64) -> !llvm.ptr
    %2044 = llvm.ptrtoint %2043 : !llvm.ptr to i64
    %2045 = llvm.sub %3, %36 : i64
    %2046 = llvm.add %2044, %2045 : i64
    %2047 = llvm.urem %2046, %3 : i64
    %2048 = llvm.sub %2046, %2047 : i64
    %2049 = llvm.inttoptr %2048 : i64 to !llvm.ptr
    %2050 = llvm.mul %38, %36 : i64
    %2051 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %2052 = llvm.ptrtoint %2051 : !llvm.ptr to i64
    %2053 = llvm.mul %2050, %2052 : i64
    "llvm.intr.memcpy"(%2049, %47, %2053) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %2054 = llvm.getelementptr %11[4] : (!llvm.ptr) -> !llvm.ptr, i32
    %2055 = llvm.ptrtoint %2054 : !llvm.ptr to i64
    %2056 = llvm.add %2055, %3 : i64
    %2057 = llvm.call @_mlir_memref_to_llvm_alloc(%2056) : (i64) -> !llvm.ptr
    %2058 = llvm.ptrtoint %2057 : !llvm.ptr to i64
    %2059 = llvm.sub %3, %36 : i64
    %2060 = llvm.add %2058, %2059 : i64
    %2061 = llvm.urem %2060, %3 : i64
    %2062 = llvm.sub %2060, %2061 : i64
    %2063 = llvm.inttoptr %2062 : i64 to !llvm.ptr
    %2064 = llvm.mul %38, %36 : i64
    %2065 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %2066 = llvm.ptrtoint %2065 : !llvm.ptr to i64
    %2067 = llvm.mul %2064, %2066 : i64
    "llvm.intr.memcpy"(%2063, %46, %2067) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %2068 = llvm.getelementptr %11[4] : (!llvm.ptr) -> !llvm.ptr, i32
    %2069 = llvm.ptrtoint %2068 : !llvm.ptr to i64
    %2070 = llvm.call @_mlir_memref_to_llvm_alloc(%2069) : (i64) -> !llvm.ptr
    %2071 = llvm.insertvalue %2070, %42[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2072 = llvm.insertvalue %2070, %2071[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2073 = llvm.insertvalue %14, %2072[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2074 = llvm.insertvalue %38, %2073[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2075 = llvm.insertvalue %36, %2074[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2076 = llvm.mul %38, %36 : i64
    %2077 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %2078 = llvm.ptrtoint %2077 : !llvm.ptr to i64
    %2079 = llvm.mul %2076, %2078 : i64
    "llvm.intr.memcpy"(%2070, %2049, %2079) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%2043) : (!llvm.ptr) -> ()
    %2080 = llvm.getelementptr %11[4] : (!llvm.ptr) -> !llvm.ptr, i32
    %2081 = llvm.ptrtoint %2080 : !llvm.ptr to i64
    %2082 = llvm.call @_mlir_memref_to_llvm_alloc(%2081) : (i64) -> !llvm.ptr
    %2083 = llvm.insertvalue %2082, %42[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2084 = llvm.insertvalue %2082, %2083[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2085 = llvm.insertvalue %14, %2084[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2086 = llvm.insertvalue %38, %2085[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2087 = llvm.insertvalue %36, %2086[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2088 = llvm.mul %38, %36 : i64
    %2089 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %2090 = llvm.ptrtoint %2089 : !llvm.ptr to i64
    %2091 = llvm.mul %2088, %2090 : i64
    "llvm.intr.memcpy"(%2082, %2063, %2091) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%2057) : (!llvm.ptr) -> ()
    llvm.br ^bb300(%18, %18, %2033, %2035, %2037, %2031, %2039, %2075, %2087 : i64, i64, i32, i32, i32, i32, i32, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)
  ^bb300(%2092: i64, %2093: i64, %2094: i32, %2095: i32, %2096: i32, %2097: i32, %2098: i32, %2099: !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, %2100: !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>):  // 2 preds: ^bb299, ^bb301
    %2101 = llvm.icmp "slt" %2092, %27 : i64
    llvm.cond_br %2101, ^bb301, ^bb302
  ^bb301:  // pred: ^bb300
    %2102 = llvm.add %2093, %28 : i64
    %2103 = llvm.extractvalue %2099[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2104 = llvm.load %2103 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %2105 = llvm.add %2094, %2095 : i32
    %2106 = llvm.shl %2095, %2104 : i32
    %2107 = llvm.icmp "ult" %2104, %26 : i32
    %2108 = llvm.select %2107, %2106, %25 : i1, i32
    %2109 = llvm.sub %26, %2104 : i32
    %2110 = llvm.lshr %2095, %2109 : i32
    %2111 = llvm.icmp "ult" %2109, %26 : i32
    %2112 = llvm.select %2111, %2110, %25 : i1, i32
    %2113 = llvm.or %2108, %2112 : i32
    %2114 = llvm.xor %2105, %2113 : i32
    %2115 = llvm.extractvalue %2099[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2116 = llvm.getelementptr inbounds %2115[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %2117 = llvm.load %2116 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %2118 = llvm.add %2105, %2114 : i32
    %2119 = llvm.shl %2114, %2117 : i32
    %2120 = llvm.icmp "ult" %2117, %26 : i32
    %2121 = llvm.select %2120, %2119, %25 : i1, i32
    %2122 = llvm.sub %26, %2117 : i32
    %2123 = llvm.lshr %2114, %2122 : i32
    %2124 = llvm.icmp "ult" %2122, %26 : i32
    %2125 = llvm.select %2124, %2123, %25 : i1, i32
    %2126 = llvm.or %2121, %2125 : i32
    %2127 = llvm.xor %2118, %2126 : i32
    %2128 = llvm.extractvalue %2099[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2129 = llvm.getelementptr inbounds %2128[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %2130 = llvm.load %2129 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %2131 = llvm.add %2118, %2127 : i32
    %2132 = llvm.shl %2127, %2130 : i32
    %2133 = llvm.icmp "ult" %2130, %26 : i32
    %2134 = llvm.select %2133, %2132, %25 : i1, i32
    %2135 = llvm.sub %26, %2130 : i32
    %2136 = llvm.lshr %2127, %2135 : i32
    %2137 = llvm.icmp "ult" %2135, %26 : i32
    %2138 = llvm.select %2137, %2136, %25 : i1, i32
    %2139 = llvm.or %2134, %2138 : i32
    %2140 = llvm.xor %2131, %2139 : i32
    %2141 = llvm.extractvalue %2099[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2142 = llvm.getelementptr inbounds %2141[3] : (!llvm.ptr) -> !llvm.ptr, i32
    %2143 = llvm.load %2142 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %2144 = llvm.add %2131, %2140 : i32
    %2145 = llvm.shl %2140, %2143 : i32
    %2146 = llvm.icmp "ult" %2143, %26 : i32
    %2147 = llvm.select %2146, %2145, %25 : i1, i32
    %2148 = llvm.sub %26, %2143 : i32
    %2149 = llvm.lshr %2140, %2148 : i32
    %2150 = llvm.icmp "ult" %2148, %26 : i32
    %2151 = llvm.select %2150, %2149, %25 : i1, i32
    %2152 = llvm.or %2147, %2151 : i32
    %2153 = llvm.xor %2144, %2152 : i32
    %2154 = llvm.add %2144, %2096 : i32
    %2155 = llvm.add %2153, %2097 : i32
    %2156 = llvm.trunc %2102 : i64 to i32
    %2157 = llvm.add %2155, %2156 : i32
    %2158 = llvm.getelementptr %11[4] : (!llvm.ptr) -> !llvm.ptr, i32
    %2159 = llvm.ptrtoint %2158 : !llvm.ptr to i64
    %2160 = llvm.add %2159, %3 : i64
    %2161 = llvm.call @_mlir_memref_to_llvm_alloc(%2160) : (i64) -> !llvm.ptr
    %2162 = llvm.ptrtoint %2161 : !llvm.ptr to i64
    %2163 = llvm.sub %3, %36 : i64
    %2164 = llvm.add %2162, %2163 : i64
    %2165 = llvm.urem %2164, %3 : i64
    %2166 = llvm.sub %2164, %2165 : i64
    %2167 = llvm.inttoptr %2166 : i64 to !llvm.ptr
    %2168 = llvm.extractvalue %2100[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2169 = llvm.mul %2168, %36 : i64
    %2170 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %2171 = llvm.ptrtoint %2170 : !llvm.ptr to i64
    %2172 = llvm.mul %2169, %2171 : i64
    %2173 = llvm.extractvalue %2100[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2174 = llvm.extractvalue %2100[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2175 = llvm.getelementptr inbounds %2173[%2174] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    "llvm.intr.memcpy"(%2167, %2175, %2172) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %2176 = llvm.extractvalue %2100[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.call @_mlir_memref_to_llvm_free(%2176) : (!llvm.ptr) -> ()
    %2177 = llvm.getelementptr %11[4] : (!llvm.ptr) -> !llvm.ptr, i32
    %2178 = llvm.ptrtoint %2177 : !llvm.ptr to i64
    %2179 = llvm.add %2178, %3 : i64
    %2180 = llvm.call @_mlir_memref_to_llvm_alloc(%2179) : (i64) -> !llvm.ptr
    %2181 = llvm.ptrtoint %2180 : !llvm.ptr to i64
    %2182 = llvm.sub %3, %36 : i64
    %2183 = llvm.add %2181, %2182 : i64
    %2184 = llvm.urem %2183, %3 : i64
    %2185 = llvm.sub %2183, %2184 : i64
    %2186 = llvm.inttoptr %2185 : i64 to !llvm.ptr
    %2187 = llvm.extractvalue %2099[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2188 = llvm.mul %2187, %36 : i64
    %2189 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %2190 = llvm.ptrtoint %2189 : !llvm.ptr to i64
    %2191 = llvm.mul %2188, %2190 : i64
    %2192 = llvm.extractvalue %2099[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2193 = llvm.extractvalue %2099[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2194 = llvm.getelementptr inbounds %2192[%2193] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    "llvm.intr.memcpy"(%2186, %2194, %2191) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %2195 = llvm.extractvalue %2099[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.call @_mlir_memref_to_llvm_free(%2195) : (!llvm.ptr) -> ()
    %2196 = llvm.getelementptr %11[4] : (!llvm.ptr) -> !llvm.ptr, i32
    %2197 = llvm.ptrtoint %2196 : !llvm.ptr to i64
    %2198 = llvm.call @_mlir_memref_to_llvm_alloc(%2197) : (i64) -> !llvm.ptr
    %2199 = llvm.insertvalue %2198, %42[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2200 = llvm.insertvalue %2198, %2199[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2201 = llvm.insertvalue %14, %2200[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2202 = llvm.insertvalue %38, %2201[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2203 = llvm.insertvalue %36, %2202[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2204 = llvm.mul %38, %36 : i64
    %2205 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %2206 = llvm.ptrtoint %2205 : !llvm.ptr to i64
    %2207 = llvm.mul %2204, %2206 : i64
    "llvm.intr.memcpy"(%2198, %2167, %2207) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%2161) : (!llvm.ptr) -> ()
    %2208 = llvm.getelementptr %11[4] : (!llvm.ptr) -> !llvm.ptr, i32
    %2209 = llvm.ptrtoint %2208 : !llvm.ptr to i64
    %2210 = llvm.call @_mlir_memref_to_llvm_alloc(%2209) : (i64) -> !llvm.ptr
    %2211 = llvm.insertvalue %2210, %42[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2212 = llvm.insertvalue %2210, %2211[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2213 = llvm.insertvalue %14, %2212[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2214 = llvm.insertvalue %38, %2213[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2215 = llvm.insertvalue %36, %2214[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2216 = llvm.mul %38, %36 : i64
    %2217 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %2218 = llvm.ptrtoint %2217 : !llvm.ptr to i64
    %2219 = llvm.mul %2216, %2218 : i64
    "llvm.intr.memcpy"(%2210, %2186, %2219) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%2180) : (!llvm.ptr) -> ()
    %2220 = llvm.add %2092, %28 : i64
    llvm.br ^bb300(%2220, %2102, %2154, %2157, %2097, %2098, %2096, %2203, %2215 : i64, i64, i32, i32, i32, i32, i32, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)
  ^bb302:  // pred: ^bb300
    %2221 = llvm.extractvalue %2100[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.call @_mlir_memref_to_llvm_free(%2221) : (!llvm.ptr) -> ()
    %2222 = llvm.extractvalue %2099[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.call @_mlir_memref_to_llvm_free(%2222) : (!llvm.ptr) -> ()
    %2223 = llvm.zext %2094 : i32 to i64
    %2224 = llvm.zext %2095 : i32 to i64
    %2225 = llvm.shl %2223, %17 : i64
    %2226 = llvm.or %2225, %2224 : i64
    %2227 = llvm.urem %2025, %19 : i64
    %2228 = llvm.mul %2227, %29 : i64
    %2229 = llvm.urem %2226, %19 : i64
    %2230 = llvm.add %2228, %2229 : i64
    %2231 = llvm.urem %2230, %19 : i64
    %2232 = llvm.load %arg10 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %2233 = llvm.load %arg13 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %2234 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %2235 = llvm.ptrtoint %2234 : !llvm.ptr to i64
    %2236 = llvm.add %2235, %3 : i64
    %2237 = llvm.call @_mlir_memref_to_llvm_alloc(%2236) : (i64) -> !llvm.ptr
    %2238 = llvm.ptrtoint %2237 : !llvm.ptr to i64
    %2239 = llvm.sub %3, %36 : i64
    %2240 = llvm.add %2238, %2239 : i64
    %2241 = llvm.urem %2240, %3 : i64
    %2242 = llvm.sub %2240, %2241 : i64
    %2243 = llvm.inttoptr %2242 : i64 to !llvm.ptr
    %2244 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %2245 = llvm.ptrtoint %2244 : !llvm.ptr to i64
    %2246 = llvm.add %2245, %3 : i64
    %2247 = llvm.call @_mlir_memref_to_llvm_alloc(%2246) : (i64) -> !llvm.ptr
    %2248 = llvm.ptrtoint %2247 : !llvm.ptr to i64
    %2249 = llvm.sub %3, %36 : i64
    %2250 = llvm.add %2248, %2249 : i64
    %2251 = llvm.urem %2250, %3 : i64
    %2252 = llvm.sub %2250, %2251 : i64
    %2253 = llvm.inttoptr %2252 : i64 to !llvm.ptr
    %2254 = llvm.getelementptr %11[96] : (!llvm.ptr) -> !llvm.ptr, f32
    %2255 = llvm.ptrtoint %2254 : !llvm.ptr to i64
    %2256 = llvm.add %2255, %3 : i64
    %2257 = llvm.call @_mlir_memref_to_llvm_alloc(%2256) : (i64) -> !llvm.ptr
    %2258 = llvm.ptrtoint %2257 : !llvm.ptr to i64
    %2259 = llvm.sub %3, %36 : i64
    %2260 = llvm.add %2258, %2259 : i64
    %2261 = llvm.urem %2260, %3 : i64
    %2262 = llvm.sub %2260, %2261 : i64
    %2263 = llvm.inttoptr %2262 : i64 to !llvm.ptr
    %2264 = llvm.getelementptr %11[8] : (!llvm.ptr) -> !llvm.ptr, f32
    %2265 = llvm.ptrtoint %2264 : !llvm.ptr to i64
    %2266 = llvm.add %2265, %3 : i64
    %2267 = llvm.call @_mlir_memref_to_llvm_alloc(%2266) : (i64) -> !llvm.ptr
    %2268 = llvm.ptrtoint %2267 : !llvm.ptr to i64
    %2269 = llvm.sub %3, %36 : i64
    %2270 = llvm.add %2268, %2269 : i64
    %2271 = llvm.urem %2270, %3 : i64
    %2272 = llvm.sub %2270, %2271 : i64
    %2273 = llvm.inttoptr %2272 : i64 to !llvm.ptr
    %2274 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %2275 = llvm.ptrtoint %2274 : !llvm.ptr to i64
    %2276 = llvm.add %2275, %3 : i64
    %2277 = llvm.call @_mlir_memref_to_llvm_alloc(%2276) : (i64) -> !llvm.ptr
    %2278 = llvm.ptrtoint %2277 : !llvm.ptr to i64
    %2279 = llvm.sub %3, %36 : i64
    %2280 = llvm.add %2278, %2279 : i64
    %2281 = llvm.urem %2280, %3 : i64
    %2282 = llvm.sub %2280, %2281 : i64
    %2283 = llvm.inttoptr %2282 : i64 to !llvm.ptr
    %2284 = llvm.getelementptr %11[96] : (!llvm.ptr) -> !llvm.ptr, f32
    %2285 = llvm.ptrtoint %2284 : !llvm.ptr to i64
    %2286 = llvm.call @_mlir_memref_to_llvm_alloc(%2285) : (i64) -> !llvm.ptr
    %2287 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %2288 = llvm.ptrtoint %2287 : !llvm.ptr to i64
    %2289 = llvm.call @_mlir_memref_to_llvm_alloc(%2288) : (i64) -> !llvm.ptr
    %2290 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %2291 = llvm.ptrtoint %2290 : !llvm.ptr to i64
    %2292 = llvm.call @_mlir_memref_to_llvm_alloc(%2291) : (i64) -> !llvm.ptr
    %2293 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %2294 = llvm.ptrtoint %2293 : !llvm.ptr to i64
    %2295 = llvm.call @_mlir_memref_to_llvm_alloc(%2294) : (i64) -> !llvm.ptr
    %2296 = llvm.getelementptr %11[96] : (!llvm.ptr) -> !llvm.ptr, f32
    %2297 = llvm.ptrtoint %2296 : !llvm.ptr to i64
    %2298 = llvm.add %2297, %3 : i64
    %2299 = llvm.call @_mlir_memref_to_llvm_alloc(%2298) : (i64) -> !llvm.ptr
    %2300 = llvm.ptrtoint %2299 : !llvm.ptr to i64
    %2301 = llvm.sub %3, %36 : i64
    %2302 = llvm.add %2300, %2301 : i64
    %2303 = llvm.urem %2302, %3 : i64
    %2304 = llvm.sub %2302, %2303 : i64
    %2305 = llvm.inttoptr %2304 : i64 to !llvm.ptr
    %2306 = llvm.getelementptr %11[96] : (!llvm.ptr) -> !llvm.ptr, f32
    %2307 = llvm.ptrtoint %2306 : !llvm.ptr to i64
    %2308 = llvm.call @_mlir_memref_to_llvm_alloc(%2307) : (i64) -> !llvm.ptr
    %2309 = llvm.insertvalue %2308, %40[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2310 = llvm.insertvalue %2308, %2309[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2311 = llvm.insertvalue %14, %2310[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2312 = llvm.insertvalue %38, %2311[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2313 = llvm.insertvalue %39, %2312[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2314 = llvm.insertvalue %15, %2313[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2315 = llvm.insertvalue %2, %2314[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2316 = llvm.insertvalue %15, %2315[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2317 = llvm.insertvalue %36, %2316[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2318 = llvm.mul %arg3, %36 : i64
    %2319 = llvm.mul %2318, %arg4 : i64
    %2320 = llvm.mul %2319, %arg5 : i64
    %2321 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %2322 = llvm.ptrtoint %2321 : !llvm.ptr to i64
    %2323 = llvm.mul %2320, %2322 : i64
    %2324 = llvm.getelementptr inbounds %arg1[%arg2] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%2308, %2324, %2323) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.br ^bb303(%14, %2317, %2232, %2233, %12, %12 : i64, !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>, f32, f32, f64, f64)
  ^bb303(%2325: i64, %2326: !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>, %2327: f32, %2328: f32, %2329: f64, %2330: f64):  // 2 preds: ^bb302, ^bb373
    %2331 = llvm.icmp "slt" %2325, %15 : i64
    llvm.cond_br %2331, ^bb304, ^bb374
  ^bb304:  // pred: ^bb303
    llvm.store %2327, %2243 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f32, !llvm.ptr
    llvm.store %2328, %2253 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f32, !llvm.ptr
    %2332 = llvm.getelementptr %11[96] : (!llvm.ptr) -> !llvm.ptr, f32
    %2333 = llvm.ptrtoint %2332 : !llvm.ptr to i64
    %2334 = llvm.add %2333, %3 : i64
    %2335 = llvm.call @_mlir_memref_to_llvm_alloc(%2334) : (i64) -> !llvm.ptr
    %2336 = llvm.ptrtoint %2335 : !llvm.ptr to i64
    %2337 = llvm.sub %3, %36 : i64
    %2338 = llvm.add %2336, %2337 : i64
    %2339 = llvm.urem %2338, %3 : i64
    %2340 = llvm.sub %2338, %2339 : i64
    %2341 = llvm.inttoptr %2340 : i64 to !llvm.ptr
    llvm.br ^bb305(%14 : i64)
  ^bb305(%2342: i64):  // 2 preds: ^bb304, ^bb312
    %2343 = llvm.icmp "slt" %2342, %38 : i64
    llvm.cond_br %2343, ^bb306, ^bb313
  ^bb306:  // pred: ^bb305
    llvm.br ^bb307(%14 : i64)
  ^bb307(%2344: i64):  // 2 preds: ^bb306, ^bb311
    %2345 = llvm.icmp "slt" %2344, %39 : i64
    llvm.cond_br %2345, ^bb308, ^bb312
  ^bb308:  // pred: ^bb307
    llvm.br ^bb309(%14 : i64)
  ^bb309(%2346: i64):  // 2 preds: ^bb308, ^bb310
    %2347 = llvm.icmp "slt" %2346, %15 : i64
    llvm.cond_br %2347, ^bb310, ^bb311
  ^bb310:  // pred: ^bb309
    %2348 = llvm.load %8 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %2349 = llvm.mul %2342, %2 : i64
    %2350 = llvm.mul %2344, %15 : i64
    %2351 = llvm.add %2349, %2350 : i64
    %2352 = llvm.add %2351, %2346 : i64
    %2353 = llvm.getelementptr inbounds %2263[%2352] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %2348, %2353 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f32, !llvm.ptr
    %2354 = llvm.add %2346, %36 : i64
    llvm.br ^bb309(%2354 : i64)
  ^bb311:  // pred: ^bb309
    %2355 = llvm.add %2344, %36 : i64
    llvm.br ^bb307(%2355 : i64)
  ^bb312:  // pred: ^bb307
    %2356 = llvm.add %2342, %36 : i64
    llvm.br ^bb305(%2356 : i64)
  ^bb313:  // pred: ^bb305
    %2357 = llvm.getelementptr %11[96] : (!llvm.ptr) -> !llvm.ptr, f32
    %2358 = llvm.ptrtoint %2357 : !llvm.ptr to i64
    %2359 = llvm.call @_mlir_memref_to_llvm_alloc(%2358) : (i64) -> !llvm.ptr
    %2360 = llvm.insertvalue %2359, %40[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2361 = llvm.insertvalue %2359, %2360[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2362 = llvm.insertvalue %14, %2361[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2363 = llvm.insertvalue %38, %2362[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2364 = llvm.insertvalue %39, %2363[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2365 = llvm.insertvalue %15, %2364[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2366 = llvm.insertvalue %2, %2365[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2367 = llvm.insertvalue %15, %2366[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2368 = llvm.insertvalue %36, %2367[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2369 = llvm.mul %38, %36 : i64
    %2370 = llvm.mul %2369, %39 : i64
    %2371 = llvm.mul %2370, %15 : i64
    %2372 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %2373 = llvm.ptrtoint %2372 : !llvm.ptr to i64
    %2374 = llvm.mul %2371, %2373 : i64
    "llvm.intr.memcpy"(%2359, %2263, %2374) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.br ^bb314(%14, %2368, %12, %12, %12 : i64, !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>, f64, f64, f64)
  ^bb314(%2375: i64, %2376: !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>, %2377: f64, %2378: f64, %2379: f64):  // 2 preds: ^bb313, ^bb327
    %2380 = llvm.icmp "slt" %2375, %16 : i64
    llvm.cond_br %2380, ^bb315, ^bb328
  ^bb315:  // pred: ^bb314
    %2381 = llvm.mul %2325, %17 : i64
    %2382 = llvm.add %2231, %2381 : i64
    %2383 = llvm.add %2382, %2375 : i64
    %2384 = llvm.srem %2383, %19 : i64
    %2385 = llvm.icmp "ne" %2384, %18 : i64
    %2386 = llvm.icmp "slt" %2384, %18 : i64
    %2387 = llvm.and %2386, %2385 : i1
    %2388 = llvm.add %2384, %19 : i64
    %2389 = llvm.select %2387, %2388, %2384 : i1, i64
    %2390 = llvm.icmp "slt" %2389, %18 : i64
    %2391 = llvm.add %2389, %19 : i64
    %2392 = llvm.select %2390, %2391, %2389 : i1, i64
    %2393 = llvm.icmp "sgt" %2392, %14 : i64
    %2394 = llvm.select %2393, %2392, %14 : i1, i64
    %2395 = llvm.icmp "slt" %2394, %20 : i64
    %2396 = llvm.select %2395, %2394, %20 : i1, i64
    %2397 = llvm.mul %2396, %39 overflow<nsw> : i64
    %2398 = llvm.getelementptr inbounds %arg31[%2396] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %2399 = llvm.load %2398 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %2400 = llvm.getelementptr inbounds %arg36[%2396] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %2401 = llvm.load %2400 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    llvm.br ^bb316(%14 : i64)
  ^bb316(%2402: i64):  // 2 preds: ^bb315, ^bb317
    %2403 = llvm.icmp "slt" %2402, %39 : i64
    llvm.cond_br %2403, ^bb317, ^bb318
  ^bb317:  // pred: ^bb316
    %2404 = llvm.getelementptr inbounds %arg24[%2397] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %2405 = llvm.getelementptr inbounds %2404[%2402] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %2406 = llvm.load %2405 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %2407 = llvm.getelementptr inbounds %2273[%2402] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %2406, %2407 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f32, !llvm.ptr
    %2408 = llvm.add %2402, %36 : i64
    llvm.br ^bb316(%2408 : i64)
  ^bb318:  // pred: ^bb316
    %2409 = llvm.extractvalue %2326[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2410 = llvm.extractvalue %2326[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2411 = llvm.extractvalue %2326[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2412 = llvm.extractvalue %2326[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2413 = llvm.extractvalue %2326[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2414 = llvm.extractvalue %2326[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2415 = llvm.extractvalue %2326[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2416 = llvm.extractvalue %2326[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2417 = llvm.extractvalue %2326[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2418 = llvm.call @qnode_forward_0(%2409, %2410, %2411, %2412, %2413, %2414, %2415, %2416, %2417, %2267, %2273, %14, %39, %36) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> !llvm.struct<(ptr, ptr, i64)>
    %2419 = llvm.extractvalue %2418[1] : !llvm.struct<(ptr, ptr, i64)> 
    %2420 = llvm.load %2419 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    %2421 = llvm.fpext %2328 : f32 to f64
    %2422 = llvm.fmul %2421, %2420 : f64
    %2423 = llvm.fpext %2327 : f32 to f64
    %2424 = llvm.fadd %2422, %2423 : f64
    %2425 = llvm.fpext %2399 : f32 to f64
    %2426 = llvm.fpext %2401 : f32 to f64
    %2427 = llvm.fcmp "ugt" %2424, %12 : f64
    %2428 = llvm.select %2427, %2424, %12 : i1, f64
    %2429 = llvm.select %1, %12, %2428 : i1, f64
    %2430 = llvm.fcmp "une" %2424, %2424 : f64
    %2431 = llvm.intr.fabs(%2424) : (f64) -> f64
    %2432 = llvm.fneg %2431 : f64
    %2433 = llvm.intr.exp(%2432) : (f64) -> f64
    %2434 = llvm.fadd %13, %2433 : f64
    %2435 = llvm.intr.log(%2434) : (f64) -> f64
    %2436 = llvm.fadd %2429, %2435 : f64
    %2437 = llvm.select %2430, %2424, %2436 : i1, f64
    %2438 = llvm.fmul %2425, %2424 : f64
    %2439 = llvm.fsub %2437, %2438 : f64
    %2440 = llvm.fmul %2426, %2439 : f64
    llvm.store %12, %2283 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    llvm.store %13, %2283 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %2441 = llvm.extractvalue %2326[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2442 = llvm.extractvalue %2326[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2443 = llvm.mul %38, %2 : i64
    %2444 = llvm.add %2443, %14 : i64
    %2445 = llvm.mul %2444, %38 : i64
    "llvm.intr.memset"(%2286, %35, %2445) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
    %2446 = llvm.extractvalue %2326[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2447 = llvm.extractvalue %2326[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2448 = llvm.extractvalue %2326[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2449 = llvm.extractvalue %2326[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2450 = llvm.extractvalue %2326[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2451 = llvm.extractvalue %2326[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2452 = llvm.extractvalue %2326[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2453 = llvm.mul %38, %36 : i64
    "llvm.intr.memset"(%2289, %35, %2453) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
    %2454 = llvm.mul %38, %36 : i64
    "llvm.intr.memset"(%2292, %35, %2454) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
    llvm.call @__enzyme_autodiff0(%32, %33, %2441, %2442, %2286, %2446, %2447, %2448, %2449, %2450, %2451, %2452, %33, %2237, %2243, %2289, %14, %33, %2247, %2253, %2292, %14, %33, %arg23, %33, %arg24, %2397, %39, %36, %33, %arg30, %33, %arg31, %2396, %33, %arg35, %33, %arg36, %2396, %33, %2295, %34, %2295, %2283, %14) vararg(!llvm.func<void (...)>) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> ()
    %2455 = llvm.load %2292 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %2456 = llvm.load %2289 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    llvm.br ^bb319(%14 : i64)
  ^bb319(%2457: i64):  // 2 preds: ^bb318, ^bb326
    %2458 = llvm.icmp "slt" %2457, %38 : i64
    llvm.cond_br %2458, ^bb320, ^bb327
  ^bb320:  // pred: ^bb319
    llvm.br ^bb321(%14 : i64)
  ^bb321(%2459: i64):  // 2 preds: ^bb320, ^bb325
    %2460 = llvm.icmp "slt" %2459, %39 : i64
    llvm.cond_br %2460, ^bb322, ^bb326
  ^bb322:  // pred: ^bb321
    llvm.br ^bb323(%14 : i64)
  ^bb323(%2461: i64):  // 2 preds: ^bb322, ^bb324
    %2462 = llvm.icmp "slt" %2461, %15 : i64
    llvm.cond_br %2462, ^bb324, ^bb325
  ^bb324:  // pred: ^bb323
    %2463 = llvm.extractvalue %2376[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2464 = llvm.mul %2457, %2 : i64
    %2465 = llvm.mul %2459, %15 : i64
    %2466 = llvm.add %2464, %2465 : i64
    %2467 = llvm.add %2466, %2461 : i64
    %2468 = llvm.getelementptr inbounds %2463[%2467] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %2469 = llvm.load %2468 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %2470 = llvm.mul %2457, %2 : i64
    %2471 = llvm.mul %2459, %15 : i64
    %2472 = llvm.add %2470, %2471 : i64
    %2473 = llvm.add %2472, %2461 : i64
    %2474 = llvm.getelementptr inbounds %2286[%2473] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %2475 = llvm.load %2474 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %2476 = llvm.fadd %2469, %2475 : f32
    %2477 = llvm.mul %2457, %2 : i64
    %2478 = llvm.mul %2459, %15 : i64
    %2479 = llvm.add %2477, %2478 : i64
    %2480 = llvm.add %2479, %2461 : i64
    %2481 = llvm.getelementptr inbounds %2341[%2480] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %2476, %2481 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f32, !llvm.ptr
    %2482 = llvm.add %2461, %36 : i64
    llvm.br ^bb323(%2482 : i64)
  ^bb325:  // pred: ^bb323
    %2483 = llvm.add %2459, %36 : i64
    llvm.br ^bb321(%2483 : i64)
  ^bb326:  // pred: ^bb321
    %2484 = llvm.add %2457, %36 : i64
    llvm.br ^bb319(%2484 : i64)
  ^bb327:  // pred: ^bb319
    %2485 = llvm.extractvalue %2376[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    llvm.call @_mlir_memref_to_llvm_free(%2485) : (!llvm.ptr) -> ()
    %2486 = llvm.fpext %2456 : f32 to f64
    %2487 = llvm.fadd %2377, %2486 : f64
    %2488 = llvm.fpext %2455 : f32 to f64
    %2489 = llvm.fadd %2378, %2488 : f64
    %2490 = llvm.fadd %2379, %2440 : f64
    %2491 = llvm.getelementptr %11[96] : (!llvm.ptr) -> !llvm.ptr, f32
    %2492 = llvm.ptrtoint %2491 : !llvm.ptr to i64
    %2493 = llvm.add %2492, %3 : i64
    %2494 = llvm.call @_mlir_memref_to_llvm_alloc(%2493) : (i64) -> !llvm.ptr
    %2495 = llvm.ptrtoint %2494 : !llvm.ptr to i64
    %2496 = llvm.sub %3, %36 : i64
    %2497 = llvm.add %2495, %2496 : i64
    %2498 = llvm.urem %2497, %3 : i64
    %2499 = llvm.sub %2497, %2498 : i64
    %2500 = llvm.inttoptr %2499 : i64 to !llvm.ptr
    %2501 = llvm.mul %38, %36 : i64
    %2502 = llvm.mul %2501, %39 : i64
    %2503 = llvm.mul %2502, %15 : i64
    %2504 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %2505 = llvm.ptrtoint %2504 : !llvm.ptr to i64
    %2506 = llvm.mul %2503, %2505 : i64
    "llvm.intr.memcpy"(%2500, %2341, %2506) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %2507 = llvm.getelementptr %11[96] : (!llvm.ptr) -> !llvm.ptr, f32
    %2508 = llvm.ptrtoint %2507 : !llvm.ptr to i64
    %2509 = llvm.call @_mlir_memref_to_llvm_alloc(%2508) : (i64) -> !llvm.ptr
    %2510 = llvm.insertvalue %2509, %40[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2511 = llvm.insertvalue %2509, %2510[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2512 = llvm.insertvalue %14, %2511[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2513 = llvm.insertvalue %38, %2512[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2514 = llvm.insertvalue %39, %2513[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2515 = llvm.insertvalue %15, %2514[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2516 = llvm.insertvalue %2, %2515[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2517 = llvm.insertvalue %15, %2516[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2518 = llvm.insertvalue %36, %2517[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2519 = llvm.mul %38, %36 : i64
    %2520 = llvm.mul %2519, %39 : i64
    %2521 = llvm.mul %2520, %15 : i64
    %2522 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %2523 = llvm.ptrtoint %2522 : !llvm.ptr to i64
    %2524 = llvm.mul %2521, %2523 : i64
    "llvm.intr.memcpy"(%2509, %2500, %2524) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%2494) : (!llvm.ptr) -> ()
    %2525 = llvm.add %2375, %36 : i64
    llvm.br ^bb314(%2525, %2518, %2487, %2489, %2490 : i64, !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>, f64, f64, f64)
  ^bb328:  // pred: ^bb314
    llvm.br ^bb329(%14 : i64)
  ^bb329(%2526: i64):  // 2 preds: ^bb328, ^bb336
    %2527 = llvm.icmp "slt" %2526, %38 : i64
    llvm.cond_br %2527, ^bb330, ^bb337
  ^bb330:  // pred: ^bb329
    llvm.br ^bb331(%14 : i64)
  ^bb331(%2528: i64):  // 2 preds: ^bb330, ^bb335
    %2529 = llvm.icmp "slt" %2528, %39 : i64
    llvm.cond_br %2529, ^bb332, ^bb336
  ^bb332:  // pred: ^bb331
    llvm.br ^bb333(%14 : i64)
  ^bb333(%2530: i64):  // 2 preds: ^bb332, ^bb334
    %2531 = llvm.icmp "slt" %2530, %15 : i64
    llvm.cond_br %2531, ^bb334, ^bb335
  ^bb334:  // pred: ^bb333
    %2532 = llvm.load %10 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %2533 = llvm.mul %2526, %2 : i64
    %2534 = llvm.mul %2528, %15 : i64
    %2535 = llvm.add %2533, %2534 : i64
    %2536 = llvm.add %2535, %2530 : i64
    %2537 = llvm.getelementptr inbounds %2341[%2536] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %2532, %2537 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f32, !llvm.ptr
    %2538 = llvm.add %2530, %36 : i64
    llvm.br ^bb333(%2538 : i64)
  ^bb335:  // pred: ^bb333
    %2539 = llvm.add %2528, %36 : i64
    llvm.br ^bb331(%2539 : i64)
  ^bb336:  // pred: ^bb331
    %2540 = llvm.add %2526, %36 : i64
    llvm.br ^bb329(%2540 : i64)
  ^bb337:  // pred: ^bb329
    llvm.br ^bb338(%14 : i64)
  ^bb338(%2541: i64):  // 2 preds: ^bb337, ^bb345
    %2542 = llvm.icmp "slt" %2541, %38 : i64
    llvm.cond_br %2542, ^bb339, ^bb346
  ^bb339:  // pred: ^bb338
    llvm.br ^bb340(%14 : i64)
  ^bb340(%2543: i64):  // 2 preds: ^bb339, ^bb344
    %2544 = llvm.icmp "slt" %2543, %39 : i64
    llvm.cond_br %2544, ^bb341, ^bb345
  ^bb341:  // pred: ^bb340
    llvm.br ^bb342(%14 : i64)
  ^bb342(%2545: i64):  // 2 preds: ^bb341, ^bb343
    %2546 = llvm.icmp "slt" %2545, %15 : i64
    llvm.cond_br %2546, ^bb343, ^bb344
  ^bb343:  // pred: ^bb342
    %2547 = llvm.extractvalue %2376[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2548 = llvm.mul %2541, %2 : i64
    %2549 = llvm.mul %2543, %15 : i64
    %2550 = llvm.add %2548, %2549 : i64
    %2551 = llvm.add %2550, %2545 : i64
    %2552 = llvm.getelementptr inbounds %2547[%2551] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %2553 = llvm.load %2552 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %2554 = llvm.mul %2541, %2 : i64
    %2555 = llvm.mul %2543, %15 : i64
    %2556 = llvm.add %2554, %2555 : i64
    %2557 = llvm.add %2556, %2545 : i64
    %2558 = llvm.getelementptr inbounds %2341[%2557] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %2559 = llvm.load %2558 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %2560 = llvm.fmul %2553, %2559 : f32
    %2561 = llvm.mul %2541, %2 : i64
    %2562 = llvm.mul %2543, %15 : i64
    %2563 = llvm.add %2561, %2562 : i64
    %2564 = llvm.add %2563, %2545 : i64
    %2565 = llvm.getelementptr inbounds %2305[%2564] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %2560, %2565 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f32, !llvm.ptr
    %2566 = llvm.add %2545, %36 : i64
    llvm.br ^bb342(%2566 : i64)
  ^bb344:  // pred: ^bb342
    %2567 = llvm.add %2543, %36 : i64
    llvm.br ^bb340(%2567 : i64)
  ^bb345:  // pred: ^bb340
    %2568 = llvm.add %2541, %36 : i64
    llvm.br ^bb338(%2568 : i64)
  ^bb346:  // pred: ^bb338
    %2569 = llvm.extractvalue %2376[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    llvm.call @_mlir_memref_to_llvm_free(%2569) : (!llvm.ptr) -> ()
    llvm.br ^bb347(%14 : i64)
  ^bb347(%2570: i64):  // 2 preds: ^bb346, ^bb354
    %2571 = llvm.icmp "slt" %2570, %38 : i64
    llvm.cond_br %2571, ^bb348, ^bb355
  ^bb348:  // pred: ^bb347
    llvm.br ^bb349(%14 : i64)
  ^bb349(%2572: i64):  // 2 preds: ^bb348, ^bb353
    %2573 = llvm.icmp "slt" %2572, %39 : i64
    llvm.cond_br %2573, ^bb350, ^bb354
  ^bb350:  // pred: ^bb349
    llvm.br ^bb351(%14 : i64)
  ^bb351(%2574: i64):  // 2 preds: ^bb350, ^bb352
    %2575 = llvm.icmp "slt" %2574, %15 : i64
    llvm.cond_br %2575, ^bb352, ^bb353
  ^bb352:  // pred: ^bb351
    %2576 = llvm.load %9 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %2577 = llvm.mul %2570, %2 : i64
    %2578 = llvm.mul %2572, %15 : i64
    %2579 = llvm.add %2577, %2578 : i64
    %2580 = llvm.add %2579, %2574 : i64
    %2581 = llvm.getelementptr inbounds %2341[%2580] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %2576, %2581 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f32, !llvm.ptr
    %2582 = llvm.add %2574, %36 : i64
    llvm.br ^bb351(%2582 : i64)
  ^bb353:  // pred: ^bb351
    %2583 = llvm.add %2572, %36 : i64
    llvm.br ^bb349(%2583 : i64)
  ^bb354:  // pred: ^bb349
    %2584 = llvm.add %2570, %36 : i64
    llvm.br ^bb347(%2584 : i64)
  ^bb355:  // pred: ^bb347
    llvm.br ^bb356(%14 : i64)
  ^bb356(%2585: i64):  // 2 preds: ^bb355, ^bb363
    %2586 = llvm.icmp "slt" %2585, %38 : i64
    llvm.cond_br %2586, ^bb357, ^bb364
  ^bb357:  // pred: ^bb356
    llvm.br ^bb358(%14 : i64)
  ^bb358(%2587: i64):  // 2 preds: ^bb357, ^bb362
    %2588 = llvm.icmp "slt" %2587, %39 : i64
    llvm.cond_br %2588, ^bb359, ^bb363
  ^bb359:  // pred: ^bb358
    llvm.br ^bb360(%14 : i64)
  ^bb360(%2589: i64):  // 2 preds: ^bb359, ^bb361
    %2590 = llvm.icmp "slt" %2589, %15 : i64
    llvm.cond_br %2590, ^bb361, ^bb362
  ^bb361:  // pred: ^bb360
    %2591 = llvm.mul %2585, %2 : i64
    %2592 = llvm.mul %2587, %15 : i64
    %2593 = llvm.add %2591, %2592 : i64
    %2594 = llvm.add %2593, %2589 : i64
    %2595 = llvm.getelementptr inbounds %2341[%2594] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %2596 = llvm.load %2595 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %2597 = llvm.mul %2585, %2 : i64
    %2598 = llvm.mul %2587, %15 : i64
    %2599 = llvm.add %2597, %2598 : i64
    %2600 = llvm.add %2599, %2589 : i64
    %2601 = llvm.getelementptr inbounds %2305[%2600] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %2602 = llvm.load %2601 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %2603 = llvm.fmul %2596, %2602 : f32
    %2604 = llvm.mul %2585, %2 : i64
    %2605 = llvm.mul %2587, %15 : i64
    %2606 = llvm.add %2604, %2605 : i64
    %2607 = llvm.add %2606, %2589 : i64
    %2608 = llvm.getelementptr inbounds %2341[%2607] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %2603, %2608 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f32, !llvm.ptr
    %2609 = llvm.add %2589, %36 : i64
    llvm.br ^bb360(%2609 : i64)
  ^bb362:  // pred: ^bb360
    %2610 = llvm.add %2587, %36 : i64
    llvm.br ^bb358(%2610 : i64)
  ^bb363:  // pred: ^bb358
    %2611 = llvm.add %2585, %36 : i64
    llvm.br ^bb356(%2611 : i64)
  ^bb364:  // pred: ^bb356
    llvm.br ^bb365(%14 : i64)
  ^bb365(%2612: i64):  // 2 preds: ^bb364, ^bb372
    %2613 = llvm.icmp "slt" %2612, %38 : i64
    llvm.cond_br %2613, ^bb366, ^bb373
  ^bb366:  // pred: ^bb365
    llvm.br ^bb367(%14 : i64)
  ^bb367(%2614: i64):  // 2 preds: ^bb366, ^bb371
    %2615 = llvm.icmp "slt" %2614, %39 : i64
    llvm.cond_br %2615, ^bb368, ^bb372
  ^bb368:  // pred: ^bb367
    llvm.br ^bb369(%14 : i64)
  ^bb369(%2616: i64):  // 2 preds: ^bb368, ^bb370
    %2617 = llvm.icmp "slt" %2616, %15 : i64
    llvm.cond_br %2617, ^bb370, ^bb371
  ^bb370:  // pred: ^bb369
    %2618 = llvm.extractvalue %2326[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2619 = llvm.mul %2612, %2 : i64
    %2620 = llvm.mul %2614, %15 : i64
    %2621 = llvm.add %2619, %2620 : i64
    %2622 = llvm.add %2621, %2616 : i64
    %2623 = llvm.getelementptr inbounds %2618[%2622] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %2624 = llvm.load %2623 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %2625 = llvm.mul %2612, %2 : i64
    %2626 = llvm.mul %2614, %15 : i64
    %2627 = llvm.add %2625, %2626 : i64
    %2628 = llvm.add %2627, %2616 : i64
    %2629 = llvm.getelementptr inbounds %2341[%2628] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %2630 = llvm.load %2629 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %2631 = llvm.fsub %2624, %2630 : f32
    %2632 = llvm.mul %2612, %2 : i64
    %2633 = llvm.mul %2614, %15 : i64
    %2634 = llvm.add %2632, %2633 : i64
    %2635 = llvm.add %2634, %2616 : i64
    %2636 = llvm.getelementptr inbounds %2341[%2635] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %2631, %2636 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f32, !llvm.ptr
    %2637 = llvm.add %2616, %36 : i64
    llvm.br ^bb369(%2637 : i64)
  ^bb371:  // pred: ^bb369
    %2638 = llvm.add %2614, %36 : i64
    llvm.br ^bb367(%2638 : i64)
  ^bb372:  // pred: ^bb367
    %2639 = llvm.add %2612, %36 : i64
    llvm.br ^bb365(%2639 : i64)
  ^bb373:  // pred: ^bb365
    %2640 = llvm.extractvalue %2326[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    llvm.call @_mlir_memref_to_llvm_free(%2640) : (!llvm.ptr) -> ()
    %2641 = llvm.fmul %2377, %24 : f64
    %2642 = llvm.fptrunc %2641 : f64 to f32
    %2643 = llvm.fmul %2642, %21 : f32
    %2644 = llvm.fsub %2327, %2643 : f32
    %2645 = llvm.fmul %2378, %24 : f64
    %2646 = llvm.fptrunc %2645 : f64 to f32
    %2647 = llvm.fmul %2646, %21 : f32
    %2648 = llvm.fsub %2328, %2647 : f32
    %2649 = llvm.fdiv %2379, %22 : f64
    %2650 = llvm.fadd %2329, %2649 : f64
    %2651 = llvm.getelementptr %11[96] : (!llvm.ptr) -> !llvm.ptr, f32
    %2652 = llvm.ptrtoint %2651 : !llvm.ptr to i64
    %2653 = llvm.call @_mlir_memref_to_llvm_alloc(%2652) : (i64) -> !llvm.ptr
    %2654 = llvm.insertvalue %2653, %40[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2655 = llvm.insertvalue %2653, %2654[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2656 = llvm.insertvalue %14, %2655[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2657 = llvm.insertvalue %38, %2656[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2658 = llvm.insertvalue %39, %2657[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2659 = llvm.insertvalue %15, %2658[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2660 = llvm.insertvalue %2, %2659[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2661 = llvm.insertvalue %15, %2660[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2662 = llvm.insertvalue %36, %2661[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2663 = llvm.mul %38, %36 : i64
    %2664 = llvm.mul %2663, %39 : i64
    %2665 = llvm.mul %2664, %15 : i64
    %2666 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %2667 = llvm.ptrtoint %2666 : !llvm.ptr to i64
    %2668 = llvm.mul %2665, %2667 : i64
    "llvm.intr.memcpy"(%2653, %2341, %2668) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%2335) : (!llvm.ptr) -> ()
    %2669 = llvm.add %2325, %36 : i64
    llvm.br ^bb303(%2669, %2662, %2644, %2648, %2650, %2649 : i64, !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>, f32, f32, f64, f64)
  ^bb374:  // pred: ^bb303
    llvm.call @_mlir_memref_to_llvm_free(%2299) : (!llvm.ptr) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%2295) : (!llvm.ptr) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%2292) : (!llvm.ptr) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%2289) : (!llvm.ptr) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%2286) : (!llvm.ptr) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%2277) : (!llvm.ptr) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%2267) : (!llvm.ptr) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%2257) : (!llvm.ptr) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%2247) : (!llvm.ptr) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%2237) : (!llvm.ptr) -> ()
    %2670 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %2671 = llvm.ptrtoint %2670 : !llvm.ptr to i64
    %2672 = llvm.add %2671, %3 : i64
    %2673 = llvm.call @_mlir_memref_to_llvm_alloc(%2672) : (i64) -> !llvm.ptr
    %2674 = llvm.ptrtoint %2673 : !llvm.ptr to i64
    %2675 = llvm.sub %3, %36 : i64
    %2676 = llvm.add %2674, %2675 : i64
    %2677 = llvm.urem %2676, %3 : i64
    %2678 = llvm.sub %2676, %2677 : i64
    %2679 = llvm.inttoptr %2678 : i64 to !llvm.ptr
    %2680 = llvm.insertvalue %2673, %41[0] : !llvm.struct<(ptr, ptr, i64)> 
    %2681 = llvm.insertvalue %2679, %2680[1] : !llvm.struct<(ptr, ptr, i64)> 
    %2682 = llvm.insertvalue %14, %2681[2] : !llvm.struct<(ptr, ptr, i64)> 
    llvm.store %2327, %2679 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f32, !llvm.ptr
    %2683 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %2684 = llvm.ptrtoint %2683 : !llvm.ptr to i64
    %2685 = llvm.add %2684, %3 : i64
    %2686 = llvm.call @_mlir_memref_to_llvm_alloc(%2685) : (i64) -> !llvm.ptr
    %2687 = llvm.ptrtoint %2686 : !llvm.ptr to i64
    %2688 = llvm.sub %3, %36 : i64
    %2689 = llvm.add %2687, %2688 : i64
    %2690 = llvm.urem %2689, %3 : i64
    %2691 = llvm.sub %2689, %2690 : i64
    %2692 = llvm.inttoptr %2691 : i64 to !llvm.ptr
    %2693 = llvm.insertvalue %2686, %41[0] : !llvm.struct<(ptr, ptr, i64)> 
    %2694 = llvm.insertvalue %2692, %2693[1] : !llvm.struct<(ptr, ptr, i64)> 
    %2695 = llvm.insertvalue %14, %2694[2] : !llvm.struct<(ptr, ptr, i64)> 
    llvm.store %2328, %2692 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f32, !llvm.ptr
    %2696 = llvm.fdiv %2329, %23 : f64
    %2697 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %2698 = llvm.ptrtoint %2697 : !llvm.ptr to i64
    %2699 = llvm.add %2698, %3 : i64
    %2700 = llvm.call @_mlir_memref_to_llvm_alloc(%2699) : (i64) -> !llvm.ptr
    %2701 = llvm.ptrtoint %2700 : !llvm.ptr to i64
    %2702 = llvm.sub %3, %36 : i64
    %2703 = llvm.add %2701, %2702 : i64
    %2704 = llvm.urem %2703, %3 : i64
    %2705 = llvm.sub %2703, %2704 : i64
    %2706 = llvm.inttoptr %2705 : i64 to !llvm.ptr
    llvm.store %2696, %2706 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %2707 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %2708 = llvm.ptrtoint %2707 : !llvm.ptr to i64
    %2709 = llvm.add %2708, %3 : i64
    %2710 = llvm.call @_mlir_memref_to_llvm_alloc(%2709) : (i64) -> !llvm.ptr
    %2711 = llvm.ptrtoint %2710 : !llvm.ptr to i64
    %2712 = llvm.sub %3, %36 : i64
    %2713 = llvm.add %2711, %2712 : i64
    %2714 = llvm.urem %2713, %3 : i64
    %2715 = llvm.sub %2713, %2714 : i64
    %2716 = llvm.inttoptr %2715 : i64 to !llvm.ptr
    llvm.store %2330, %2716 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %2717 = llvm.getelementptr %11[2] : (!llvm.ptr) -> !llvm.ptr, f64
    %2718 = llvm.ptrtoint %2717 : !llvm.ptr to i64
    %2719 = llvm.add %2718, %3 : i64
    %2720 = llvm.call @_mlir_memref_to_llvm_alloc(%2719) : (i64) -> !llvm.ptr
    %2721 = llvm.ptrtoint %2720 : !llvm.ptr to i64
    %2722 = llvm.sub %3, %36 : i64
    %2723 = llvm.add %2721, %2722 : i64
    %2724 = llvm.urem %2723, %3 : i64
    %2725 = llvm.sub %2723, %2724 : i64
    %2726 = llvm.inttoptr %2725 : i64 to !llvm.ptr
    %2727 = llvm.insertvalue %2720, %42[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2728 = llvm.insertvalue %2726, %2727[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2729 = llvm.insertvalue %14, %2728[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2730 = llvm.insertvalue %37, %2729[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2731 = llvm.insertvalue %36, %2730[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb375(%14 : i64)
  ^bb375(%2732: i64):  // 2 preds: ^bb374, ^bb380
    %2733 = llvm.icmp "slt" %2732, %37 : i64
    llvm.cond_br %2733, ^bb376, ^bb381
  ^bb376:  // pred: ^bb375
    %2734 = llvm.icmp "ult" %2732, %36 : i64
    llvm.cond_br %2734, ^bb377, ^bb378
  ^bb377:  // pred: ^bb376
    %2735 = llvm.getelementptr inbounds %2706[%2732] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %2736 = llvm.load %2735 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.br ^bb379(%2736 : f64)
  ^bb378:  // pred: ^bb376
    %2737 = llvm.sub %2732, %36 : i64
    %2738 = llvm.getelementptr inbounds %2716[%2737] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %2739 = llvm.load %2738 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.br ^bb379(%2739 : f64)
  ^bb379(%2740: f64):  // 2 preds: ^bb377, ^bb378
    llvm.br ^bb380
  ^bb380:  // pred: ^bb379
    %2741 = llvm.getelementptr inbounds %2726[%2732] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %2740, %2741 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %2742 = llvm.add %2732, %36 : i64
    llvm.br ^bb375(%2742 : i64)
  ^bb381:  // pred: ^bb375
    llvm.call @_mlir_memref_to_llvm_free(%2710) : (!llvm.ptr) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%2700) : (!llvm.ptr) -> ()
    %2743 = llvm.extractvalue %2326[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2744 = llvm.ptrtoint %2743 : !llvm.ptr to i64
    %2745 = llvm.icmp "eq" %31, %2744 : i64
    llvm.cond_br %2745, ^bb382, ^bb383
  ^bb382:  // pred: ^bb381
    %2746 = llvm.getelementptr %11[96] : (!llvm.ptr) -> !llvm.ptr, f32
    %2747 = llvm.ptrtoint %2746 : !llvm.ptr to i64
    %2748 = llvm.call @_mlir_memref_to_llvm_alloc(%2747) : (i64) -> !llvm.ptr
    %2749 = llvm.insertvalue %2748, %40[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2750 = llvm.insertvalue %2748, %2749[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2751 = llvm.insertvalue %14, %2750[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2752 = llvm.insertvalue %38, %2751[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2753 = llvm.insertvalue %39, %2752[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2754 = llvm.insertvalue %15, %2753[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2755 = llvm.insertvalue %2, %2754[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2756 = llvm.insertvalue %15, %2755[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2757 = llvm.insertvalue %36, %2756[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2758 = llvm.extractvalue %2326[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2759 = llvm.mul %2758, %36 : i64
    %2760 = llvm.extractvalue %2326[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2761 = llvm.mul %2759, %2760 : i64
    %2762 = llvm.extractvalue %2326[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2763 = llvm.mul %2761, %2762 : i64
    %2764 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %2765 = llvm.ptrtoint %2764 : !llvm.ptr to i64
    %2766 = llvm.mul %2763, %2765 : i64
    %2767 = llvm.extractvalue %2326[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2768 = llvm.extractvalue %2326[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2769 = llvm.getelementptr inbounds %2767[%2768] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%2748, %2769, %2766) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.br ^bb384(%2757 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>)
  ^bb383:  // pred: ^bb381
    llvm.br ^bb384(%2326 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>)
  ^bb384(%2770: !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>):  // 2 preds: ^bb382, ^bb383
    llvm.br ^bb385
  ^bb385:  // pred: ^bb384
    %2771 = llvm.ptrtoint %2673 : !llvm.ptr to i64
    %2772 = llvm.icmp "eq" %31, %2771 : i64
    llvm.cond_br %2772, ^bb386, ^bb387
  ^bb386:  // pred: ^bb385
    %2773 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %2774 = llvm.ptrtoint %2773 : !llvm.ptr to i64
    %2775 = llvm.call @_mlir_memref_to_llvm_alloc(%2774) : (i64) -> !llvm.ptr
    %2776 = llvm.insertvalue %2775, %41[0] : !llvm.struct<(ptr, ptr, i64)> 
    %2777 = llvm.insertvalue %2775, %2776[1] : !llvm.struct<(ptr, ptr, i64)> 
    %2778 = llvm.insertvalue %14, %2777[2] : !llvm.struct<(ptr, ptr, i64)> 
    %2779 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %2780 = llvm.ptrtoint %2779 : !llvm.ptr to i64
    %2781 = llvm.mul %2780, %36 : i64
    "llvm.intr.memcpy"(%2775, %2679, %2781) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.br ^bb388(%2778 : !llvm.struct<(ptr, ptr, i64)>)
  ^bb387:  // pred: ^bb385
    llvm.br ^bb388(%2682 : !llvm.struct<(ptr, ptr, i64)>)
  ^bb388(%2782: !llvm.struct<(ptr, ptr, i64)>):  // 2 preds: ^bb386, ^bb387
    llvm.br ^bb389
  ^bb389:  // pred: ^bb388
    %2783 = llvm.ptrtoint %2686 : !llvm.ptr to i64
    %2784 = llvm.icmp "eq" %31, %2783 : i64
    llvm.cond_br %2784, ^bb390, ^bb391
  ^bb390:  // pred: ^bb389
    %2785 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %2786 = llvm.ptrtoint %2785 : !llvm.ptr to i64
    %2787 = llvm.call @_mlir_memref_to_llvm_alloc(%2786) : (i64) -> !llvm.ptr
    %2788 = llvm.insertvalue %2787, %41[0] : !llvm.struct<(ptr, ptr, i64)> 
    %2789 = llvm.insertvalue %2787, %2788[1] : !llvm.struct<(ptr, ptr, i64)> 
    %2790 = llvm.insertvalue %14, %2789[2] : !llvm.struct<(ptr, ptr, i64)> 
    %2791 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %2792 = llvm.ptrtoint %2791 : !llvm.ptr to i64
    %2793 = llvm.mul %2792, %36 : i64
    "llvm.intr.memcpy"(%2787, %2692, %2793) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.br ^bb392(%2790 : !llvm.struct<(ptr, ptr, i64)>)
  ^bb391:  // pred: ^bb389
    llvm.br ^bb392(%2695 : !llvm.struct<(ptr, ptr, i64)>)
  ^bb392(%2794: !llvm.struct<(ptr, ptr, i64)>):  // 2 preds: ^bb390, ^bb391
    llvm.br ^bb393
  ^bb393:  // pred: ^bb392
    %2795 = llvm.ptrtoint %arg15 : !llvm.ptr to i64
    %2796 = llvm.icmp "eq" %31, %2795 : i64
    llvm.cond_br %2796, ^bb394, ^bb395
  ^bb394:  // pred: ^bb393
    %2797 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %2798 = llvm.ptrtoint %2797 : !llvm.ptr to i64
    %2799 = llvm.call @_mlir_memref_to_llvm_alloc(%2798) : (i64) -> !llvm.ptr
    %2800 = llvm.insertvalue %2799, %41[0] : !llvm.struct<(ptr, ptr, i64)> 
    %2801 = llvm.insertvalue %2799, %2800[1] : !llvm.struct<(ptr, ptr, i64)> 
    %2802 = llvm.insertvalue %14, %2801[2] : !llvm.struct<(ptr, ptr, i64)> 
    %2803 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %2804 = llvm.ptrtoint %2803 : !llvm.ptr to i64
    %2805 = llvm.mul %2804, %36 : i64
    %2806 = llvm.getelementptr inbounds %arg16[%arg17] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    "llvm.intr.memcpy"(%2799, %2806, %2805) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.br ^bb396(%2802 : !llvm.struct<(ptr, ptr, i64)>)
  ^bb395:  // pred: ^bb393
    llvm.br ^bb396(%45 : !llvm.struct<(ptr, ptr, i64)>)
  ^bb396(%2807: !llvm.struct<(ptr, ptr, i64)>):  // 2 preds: ^bb394, ^bb395
    llvm.br ^bb397
  ^bb397:  // pred: ^bb396
    %2808 = llvm.ptrtoint %907 : !llvm.ptr to i64
    %2809 = llvm.icmp "eq" %31, %2808 : i64
    llvm.cond_br %2809, ^bb398, ^bb399
  ^bb398:  // pred: ^bb397
    %2810 = llvm.getelementptr %11[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %2811 = llvm.ptrtoint %2810 : !llvm.ptr to i64
    %2812 = llvm.call @_mlir_memref_to_llvm_alloc(%2811) : (i64) -> !llvm.ptr
    %2813 = llvm.insertvalue %2812, %42[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2814 = llvm.insertvalue %2812, %2813[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2815 = llvm.insertvalue %14, %2814[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2816 = llvm.insertvalue %37, %2815[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2817 = llvm.insertvalue %36, %2816[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2818 = llvm.mul %37, %36 : i64
    %2819 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %2820 = llvm.ptrtoint %2819 : !llvm.ptr to i64
    %2821 = llvm.mul %2818, %2820 : i64
    "llvm.intr.memcpy"(%2812, %913, %2821) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.br ^bb400(%2817 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)
  ^bb399:  // pred: ^bb397
    llvm.br ^bb400(%938 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)
  ^bb400(%2822: !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>):  // 2 preds: ^bb398, ^bb399
    llvm.br ^bb401
  ^bb401:  // pred: ^bb400
    %2823 = llvm.ptrtoint %2720 : !llvm.ptr to i64
    %2824 = llvm.icmp "eq" %31, %2823 : i64
    llvm.cond_br %2824, ^bb402, ^bb403
  ^bb402:  // pred: ^bb401
    %2825 = llvm.getelementptr %11[2] : (!llvm.ptr) -> !llvm.ptr, f64
    %2826 = llvm.ptrtoint %2825 : !llvm.ptr to i64
    %2827 = llvm.call @_mlir_memref_to_llvm_alloc(%2826) : (i64) -> !llvm.ptr
    %2828 = llvm.insertvalue %2827, %42[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2829 = llvm.insertvalue %2827, %2828[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2830 = llvm.insertvalue %14, %2829[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2831 = llvm.insertvalue %37, %2830[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2832 = llvm.insertvalue %36, %2831[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2833 = llvm.mul %37, %36 : i64
    %2834 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %2835 = llvm.ptrtoint %2834 : !llvm.ptr to i64
    %2836 = llvm.mul %2833, %2835 : i64
    "llvm.intr.memcpy"(%2827, %2726, %2836) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.br ^bb404(%2832 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)
  ^bb403:  // pred: ^bb401
    llvm.br ^bb404(%2731 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)
  ^bb404(%2837: !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>):  // 2 preds: ^bb402, ^bb403
    llvm.br ^bb405
  ^bb405:  // pred: ^bb404
    %2838 = llvm.insertvalue %2770, %0[0] : !llvm.struct<(struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>, struct<(ptr, ptr, i64)>, struct<(ptr, ptr, i64)>, struct<(ptr, ptr, i64)>, struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)> 
    %2839 = llvm.insertvalue %2782, %2838[1] : !llvm.struct<(struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>, struct<(ptr, ptr, i64)>, struct<(ptr, ptr, i64)>, struct<(ptr, ptr, i64)>, struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)> 
    %2840 = llvm.insertvalue %2794, %2839[2] : !llvm.struct<(struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>, struct<(ptr, ptr, i64)>, struct<(ptr, ptr, i64)>, struct<(ptr, ptr, i64)>, struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)> 
    %2841 = llvm.insertvalue %2807, %2840[3] : !llvm.struct<(struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>, struct<(ptr, ptr, i64)>, struct<(ptr, ptr, i64)>, struct<(ptr, ptr, i64)>, struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)> 
    %2842 = llvm.insertvalue %2822, %2841[4] : !llvm.struct<(struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>, struct<(ptr, ptr, i64)>, struct<(ptr, ptr, i64)>, struct<(ptr, ptr, i64)>, struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)> 
    %2843 = llvm.insertvalue %2837, %2842[5] : !llvm.struct<(struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>, struct<(ptr, ptr, i64)>, struct<(ptr, ptr, i64)>, struct<(ptr, ptr, i64)>, struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)> 
    llvm.return %2843 : !llvm.struct<(struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>, struct<(ptr, ptr, i64)>, struct<(ptr, ptr, i64)>, struct<(ptr, ptr, i64)>, struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>
  }
  llvm.func @_catalyst_pyface_jit_train_epoch_compiled(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
    %0 = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<(ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr)>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr)> 
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr)> 
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr)> 
    %4 = llvm.extractvalue %0[3] : !llvm.struct<(ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr)> 
    %5 = llvm.extractvalue %0[4] : !llvm.struct<(ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr)> 
    %6 = llvm.extractvalue %0[5] : !llvm.struct<(ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr)> 
    %7 = llvm.extractvalue %0[6] : !llvm.struct<(ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr)> 
    %8 = llvm.extractvalue %0[7] : !llvm.struct<(ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr)> 
    llvm.call @_catalyst_ciface_jit_train_epoch_compiled(%arg0, %1, %2, %3, %4, %5, %6, %7, %8) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }
  llvm.func @_catalyst_ciface_jit_train_epoch_compiled(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: !llvm.ptr, %arg6: !llvm.ptr, %arg7: !llvm.ptr, %arg8: !llvm.ptr) attributes {llvm.copy_memref, llvm.emit_c_interface} {
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
    %10 = llvm.load %arg2 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64)>
    %11 = llvm.extractvalue %10[0] : !llvm.struct<(ptr, ptr, i64)> 
    %12 = llvm.extractvalue %10[1] : !llvm.struct<(ptr, ptr, i64)> 
    %13 = llvm.extractvalue %10[2] : !llvm.struct<(ptr, ptr, i64)> 
    %14 = llvm.load %arg3 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64)>
    %15 = llvm.extractvalue %14[0] : !llvm.struct<(ptr, ptr, i64)> 
    %16 = llvm.extractvalue %14[1] : !llvm.struct<(ptr, ptr, i64)> 
    %17 = llvm.extractvalue %14[2] : !llvm.struct<(ptr, ptr, i64)> 
    %18 = llvm.load %arg4 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64)>
    %19 = llvm.extractvalue %18[0] : !llvm.struct<(ptr, ptr, i64)> 
    %20 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64)> 
    %21 = llvm.extractvalue %18[2] : !llvm.struct<(ptr, ptr, i64)> 
    %22 = llvm.load %arg5 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %23 = llvm.extractvalue %22[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %24 = llvm.extractvalue %22[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %25 = llvm.extractvalue %22[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %26 = llvm.extractvalue %22[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %27 = llvm.extractvalue %22[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %28 = llvm.load %arg6 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %29 = llvm.extractvalue %28[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %30 = llvm.extractvalue %28[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %31 = llvm.extractvalue %28[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %32 = llvm.extractvalue %28[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %33 = llvm.extractvalue %28[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %34 = llvm.extractvalue %28[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %35 = llvm.extractvalue %28[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %36 = llvm.load %arg7 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %37 = llvm.extractvalue %36[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %38 = llvm.extractvalue %36[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %39 = llvm.extractvalue %36[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %40 = llvm.extractvalue %36[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %41 = llvm.extractvalue %36[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %42 = llvm.load %arg8 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %43 = llvm.extractvalue %42[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %44 = llvm.extractvalue %42[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %45 = llvm.extractvalue %42[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %46 = llvm.extractvalue %42[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %47 = llvm.extractvalue %42[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %48 = llvm.call @jit_train_epoch_compiled(%1, %2, %3, %4, %5, %6, %7, %8, %9, %11, %12, %13, %15, %16, %17, %19, %20, %21, %23, %24, %25, %26, %27, %29, %30, %31, %32, %33, %34, %35, %37, %38, %39, %40, %41, %43, %44, %45, %46, %47) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> !llvm.struct<(struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>, struct<(ptr, ptr, i64)>, struct<(ptr, ptr, i64)>, struct<(ptr, ptr, i64)>, struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>
    llvm.store %48, %arg0 : !llvm.struct<(struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>, struct<(ptr, ptr, i64)>, struct<(ptr, ptr, i64)>, struct<(ptr, ptr, i64)>, struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, !llvm.ptr
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
    %9 = llvm.mlir.addressof @"{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}" : !llvm.ptr
    %10 = llvm.mlir.addressof @LightningSimulator : !llvm.ptr
    %11 = llvm.mlir.addressof @"/home/pichau/QAgents/.venv/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.so" : !llvm.ptr
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
    %70 = llvm.getelementptr inbounds %11[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<107 x i8>
    %71 = llvm.getelementptr inbounds %10[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<19 x i8>
    %72 = llvm.getelementptr inbounds %9[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<54 x i8>
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
    %10 = llvm.mlir.addressof @"{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}" : !llvm.ptr
    %11 = llvm.mlir.addressof @LightningSimulator : !llvm.ptr
    %12 = llvm.mlir.addressof @"/home/pichau/QAgents/.venv/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.so" : !llvm.ptr
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
    %70 = llvm.getelementptr inbounds %12[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<107 x i8>
    %71 = llvm.getelementptr inbounds %11[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<19 x i8>
    %72 = llvm.getelementptr inbounds %10[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<54 x i8>
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
    %10 = llvm.mlir.addressof @"{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}" : !llvm.ptr
    %11 = llvm.mlir.addressof @LightningSimulator : !llvm.ptr
    %12 = llvm.mlir.addressof @"/home/pichau/QAgents/.venv/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.so" : !llvm.ptr
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
    %21 = llvm.getelementptr inbounds %12[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<107 x i8>
    %22 = llvm.getelementptr inbounds %11[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<19 x i8>
    %23 = llvm.getelementptr inbounds %10[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<54 x i8>
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
  llvm.func @qnode_forward_0.preprocess(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: !llvm.ptr, %arg10: !llvm.ptr, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: i64) -> !llvm.struct<(ptr, ptr, i64)> attributes {sym_visibility = "private"} {
    %0 = llvm.mlir.constant(64 : index) : i64
    %1 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64)>
    %2 = llvm.mlir.addressof @__constant_xf32 : !llvm.ptr
    %3 = llvm.mlir.zero : !llvm.ptr
    %4 = llvm.mlir.constant(0 : i64) : i64
    %5 = llvm.mlir.constant(1 : i64) : i64
    %6 = llvm.mlir.constant(0 : index) : i64
    %7 = llvm.mlir.constant(8 : index) : i64
    %8 = llvm.mlir.constant(1 : index) : i64
    %9 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %10 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %11 = llvm.insertvalue %arg9, %10[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %12 = llvm.insertvalue %arg10, %11[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %13 = llvm.insertvalue %arg11, %12[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %14 = llvm.insertvalue %arg12, %13[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %15 = llvm.insertvalue %arg13, %14[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %16 = llvm.insertvalue %arg0, %9[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %17 = llvm.insertvalue %arg1, %16[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %18 = llvm.insertvalue %arg2, %17[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %19 = llvm.insertvalue %arg3, %18[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %20 = llvm.insertvalue %arg6, %19[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %21 = llvm.insertvalue %arg4, %20[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %22 = llvm.insertvalue %arg7, %21[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %23 = llvm.insertvalue %arg5, %22[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %24 = llvm.insertvalue %arg8, %23[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %25 = llvm.alloca %5 x !llvm.struct<(ptr, ptr, i64)> : (i64) -> !llvm.ptr
    %26 = llvm.alloca %5 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    %27 = llvm.alloca %5 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    %28 = llvm.alloca %5 x !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> : (i64) -> !llvm.ptr
    %29 = llvm.getelementptr %3[%arg14] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %30 = llvm.ptrtoint %29 : !llvm.ptr to i64
    %31 = llvm.call @_mlir_memref_to_llvm_alloc(%30) : (i64) -> !llvm.ptr
    %32 = llvm.insertvalue %31, %10[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %33 = llvm.insertvalue %31, %32[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %34 = llvm.insertvalue %6, %33[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %35 = llvm.insertvalue %arg14, %34[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %36 = llvm.insertvalue %8, %35[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %37 = llvm.alloca %8 x i64 : (i64) -> !llvm.ptr
    llvm.store %4, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
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
    %62 = llvm.getelementptr %3[8] : (!llvm.ptr) -> !llvm.ptr, f32
    %63 = llvm.ptrtoint %62 : !llvm.ptr to i64
    %64 = llvm.add %63, %0 : i64
    %65 = llvm.call @_mlir_memref_to_llvm_alloc(%64) : (i64) -> !llvm.ptr
    %66 = llvm.ptrtoint %65 : !llvm.ptr to i64
    %67 = llvm.sub %0, %8 : i64
    %68 = llvm.add %66, %67 : i64
    %69 = llvm.urem %68, %0 : i64
    %70 = llvm.sub %68, %69 : i64
    %71 = llvm.inttoptr %70 : i64 to !llvm.ptr
    llvm.br ^bb1(%6 : i64)
  ^bb1(%72: i64):  // 2 preds: ^bb0, ^bb2
    %73 = llvm.icmp "slt" %72, %7 : i64
    llvm.cond_br %73, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %74 = llvm.load %2 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %75 = llvm.getelementptr inbounds %71[%72] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %74, %75 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f32, !llvm.ptr
    %76 = llvm.add %72, %8 : i64
    llvm.br ^bb1(%76 : i64)
  ^bb3:  // pred: ^bb1
    llvm.br ^bb4(%6 : i64)
  ^bb4(%77: i64):  // 2 preds: ^bb3, ^bb5
    %78 = llvm.icmp "slt" %77, %7 : i64
    llvm.cond_br %78, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %79 = llvm.getelementptr inbounds %71[%77] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %80 = llvm.load %79 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %81 = llvm.getelementptr inbounds %arg10[%77] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %82 = llvm.load %81 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %83 = llvm.fmul %80, %82 : f32
    %84 = llvm.getelementptr inbounds %71[%77] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %83, %84 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f32, !llvm.ptr
    %85 = llvm.add %77, %8 : i64
    llvm.br ^bb4(%85 : i64)
  ^bb6:  // pred: ^bb4
    %86 = llvm.getelementptr inbounds %71[7] : (!llvm.ptr) -> !llvm.ptr, f32
    %87 = llvm.load %86 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %88 = llvm.fpext %87 : f32 to f64
    %89 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %90 = llvm.getelementptr inbounds %31[%89] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %88, %90 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %91 = llvm.add %89, %5 : i64
    llvm.store %91, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %92 = llvm.fpext %61 : f32 to f64
    %93 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %94 = llvm.getelementptr inbounds %31[%93] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %92, %94 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %95 = llvm.add %93, %5 : i64
    llvm.store %95, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %96 = llvm.fpext %59 : f32 to f64
    %97 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %98 = llvm.getelementptr inbounds %31[%97] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %96, %98 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %99 = llvm.add %97, %5 : i64
    llvm.store %99, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %100 = llvm.fpext %57 : f32 to f64
    %101 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %102 = llvm.getelementptr inbounds %31[%101] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %100, %102 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %103 = llvm.add %101, %5 : i64
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
    %115 = llvm.add %113, %5 : i64
    llvm.store %115, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %116 = llvm.fpext %109 : f32 to f64
    %117 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %118 = llvm.getelementptr inbounds %31[%117] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %116, %118 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %119 = llvm.add %117, %5 : i64
    llvm.store %119, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %120 = llvm.fpext %107 : f32 to f64
    %121 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %122 = llvm.getelementptr inbounds %31[%121] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %120, %122 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %123 = llvm.add %121, %5 : i64
    llvm.store %123, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %124 = llvm.fpext %105 : f32 to f64
    %125 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %126 = llvm.getelementptr inbounds %31[%125] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %124, %126 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %127 = llvm.add %125, %5 : i64
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
    %139 = llvm.add %137, %5 : i64
    llvm.store %139, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %140 = llvm.fpext %133 : f32 to f64
    %141 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %142 = llvm.getelementptr inbounds %31[%141] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %140, %142 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %143 = llvm.add %141, %5 : i64
    llvm.store %143, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %144 = llvm.fpext %131 : f32 to f64
    %145 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %146 = llvm.getelementptr inbounds %31[%145] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %144, %146 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %147 = llvm.add %145, %5 : i64
    llvm.store %147, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %148 = llvm.fpext %129 : f32 to f64
    %149 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %150 = llvm.getelementptr inbounds %31[%149] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %148, %150 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %151 = llvm.add %149, %5 : i64
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
    %163 = llvm.add %161, %5 : i64
    llvm.store %163, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %164 = llvm.fpext %157 : f32 to f64
    %165 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %166 = llvm.getelementptr inbounds %31[%165] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %164, %166 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %167 = llvm.add %165, %5 : i64
    llvm.store %167, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %168 = llvm.fpext %155 : f32 to f64
    %169 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %170 = llvm.getelementptr inbounds %31[%169] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %168, %170 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %171 = llvm.add %169, %5 : i64
    llvm.store %171, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %172 = llvm.fpext %153 : f32 to f64
    %173 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %174 = llvm.getelementptr inbounds %31[%173] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %172, %174 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %175 = llvm.add %173, %5 : i64
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
    %187 = llvm.add %185, %5 : i64
    llvm.store %187, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %188 = llvm.fpext %181 : f32 to f64
    %189 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %190 = llvm.getelementptr inbounds %31[%189] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %188, %190 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %191 = llvm.add %189, %5 : i64
    llvm.store %191, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %192 = llvm.fpext %179 : f32 to f64
    %193 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %194 = llvm.getelementptr inbounds %31[%193] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %192, %194 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %195 = llvm.add %193, %5 : i64
    llvm.store %195, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %196 = llvm.fpext %177 : f32 to f64
    %197 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %198 = llvm.getelementptr inbounds %31[%197] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %196, %198 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %199 = llvm.add %197, %5 : i64
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
    %211 = llvm.add %209, %5 : i64
    llvm.store %211, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %212 = llvm.fpext %205 : f32 to f64
    %213 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %214 = llvm.getelementptr inbounds %31[%213] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %212, %214 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %215 = llvm.add %213, %5 : i64
    llvm.store %215, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %216 = llvm.fpext %203 : f32 to f64
    %217 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %218 = llvm.getelementptr inbounds %31[%217] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %216, %218 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %219 = llvm.add %217, %5 : i64
    llvm.store %219, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %220 = llvm.fpext %201 : f32 to f64
    %221 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %222 = llvm.getelementptr inbounds %31[%221] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %220, %222 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %223 = llvm.add %221, %5 : i64
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
    %233 = llvm.add %231, %5 : i64
    llvm.store %233, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %234 = llvm.fpext %228 : f32 to f64
    %235 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %236 = llvm.getelementptr inbounds %31[%235] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %234, %236 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %237 = llvm.add %235, %5 : i64
    llvm.store %237, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %238 = llvm.fpext %227 : f32 to f64
    %239 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %240 = llvm.getelementptr inbounds %31[%239] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %238, %240 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %241 = llvm.add %239, %5 : i64
    llvm.store %241, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %242 = llvm.fpext %225 : f32 to f64
    %243 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %244 = llvm.getelementptr inbounds %31[%243] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %242, %244 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %245 = llvm.add %243, %5 : i64
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
    %257 = llvm.add %255, %5 : i64
    llvm.store %257, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %258 = llvm.fpext %251 : f32 to f64
    %259 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %260 = llvm.getelementptr inbounds %31[%259] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %258, %260 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %261 = llvm.add %259, %5 : i64
    llvm.store %261, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %262 = llvm.fpext %249 : f32 to f64
    %263 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %264 = llvm.getelementptr inbounds %31[%263] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %262, %264 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %265 = llvm.add %263, %5 : i64
    llvm.store %265, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %266 = llvm.fpext %247 : f32 to f64
    %267 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %268 = llvm.getelementptr inbounds %31[%267] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %266, %268 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %269 = llvm.add %267, %5 : i64
    llvm.store %269, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %270 = llvm.fpext %55 : f32 to f64
    %271 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %272 = llvm.getelementptr inbounds %31[%271] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %270, %272 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %273 = llvm.add %271, %5 : i64
    llvm.store %273, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %274 = llvm.fpext %53 : f32 to f64
    %275 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %276 = llvm.getelementptr inbounds %31[%275] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %274, %276 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %277 = llvm.add %275, %5 : i64
    llvm.store %277, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %278 = llvm.fpext %51 : f32 to f64
    %279 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %280 = llvm.getelementptr inbounds %31[%279] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %278, %280 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %281 = llvm.add %279, %5 : i64
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
    %291 = llvm.add %289, %5 : i64
    llvm.store %291, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %292 = llvm.fpext %285 : f32 to f64
    %293 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %294 = llvm.getelementptr inbounds %31[%293] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %292, %294 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %295 = llvm.add %293, %5 : i64
    llvm.store %295, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %296 = llvm.fpext %283 : f32 to f64
    %297 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %298 = llvm.getelementptr inbounds %31[%297] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %296, %298 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %299 = llvm.add %297, %5 : i64
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
    %309 = llvm.add %307, %5 : i64
    llvm.store %309, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %310 = llvm.fpext %303 : f32 to f64
    %311 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %312 = llvm.getelementptr inbounds %31[%311] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %310, %312 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %313 = llvm.add %311, %5 : i64
    llvm.store %313, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %314 = llvm.fpext %301 : f32 to f64
    %315 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %316 = llvm.getelementptr inbounds %31[%315] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %314, %316 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %317 = llvm.add %315, %5 : i64
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
    %327 = llvm.add %325, %5 : i64
    llvm.store %327, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %328 = llvm.fpext %321 : f32 to f64
    %329 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %330 = llvm.getelementptr inbounds %31[%329] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %328, %330 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %331 = llvm.add %329, %5 : i64
    llvm.store %331, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %332 = llvm.fpext %319 : f32 to f64
    %333 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %334 = llvm.getelementptr inbounds %31[%333] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %332, %334 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %335 = llvm.add %333, %5 : i64
    llvm.store %335, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %336 = llvm.fpext %49 : f32 to f64
    %337 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %338 = llvm.getelementptr inbounds %31[%337] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %336, %338 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %339 = llvm.add %337, %5 : i64
    llvm.store %339, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %340 = llvm.fpext %47 : f32 to f64
    %341 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %342 = llvm.getelementptr inbounds %31[%341] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %340, %342 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %343 = llvm.add %341, %5 : i64
    llvm.store %343, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %344 = llvm.fpext %45 : f32 to f64
    %345 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %346 = llvm.getelementptr inbounds %31[%345] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %344, %346 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %347 = llvm.add %345, %5 : i64
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
    %363 = llvm.add %361, %5 : i64
    llvm.store %363, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %364 = llvm.fpext %357 : f32 to f64
    %365 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %366 = llvm.getelementptr inbounds %31[%365] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %364, %366 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %367 = llvm.add %365, %5 : i64
    llvm.store %367, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %368 = llvm.fpext %355 : f32 to f64
    %369 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %370 = llvm.getelementptr inbounds %31[%369] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %368, %370 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %371 = llvm.add %369, %5 : i64
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
    %381 = llvm.add %379, %5 : i64
    llvm.store %381, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %382 = llvm.fpext %375 : f32 to f64
    %383 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %384 = llvm.getelementptr inbounds %31[%383] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %382, %384 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %385 = llvm.add %383, %5 : i64
    llvm.store %385, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %386 = llvm.fpext %373 : f32 to f64
    %387 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %388 = llvm.getelementptr inbounds %31[%387] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %386, %388 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %389 = llvm.add %387, %5 : i64
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
    %399 = llvm.add %397, %5 : i64
    llvm.store %399, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %400 = llvm.fpext %393 : f32 to f64
    %401 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %402 = llvm.getelementptr inbounds %31[%401] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %400, %402 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %403 = llvm.add %401, %5 : i64
    llvm.store %403, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %404 = llvm.fpext %391 : f32 to f64
    %405 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %406 = llvm.getelementptr inbounds %31[%405] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %404, %406 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %407 = llvm.add %405, %5 : i64
    llvm.store %407, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %408 = llvm.fpext %353 : f32 to f64
    %409 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %410 = llvm.getelementptr inbounds %31[%409] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %408, %410 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %411 = llvm.add %409, %5 : i64
    llvm.store %411, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %412 = llvm.fpext %351 : f32 to f64
    %413 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %414 = llvm.getelementptr inbounds %31[%413] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %412, %414 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %415 = llvm.add %413, %5 : i64
    llvm.store %415, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %416 = llvm.fpext %349 : f32 to f64
    %417 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %418 = llvm.getelementptr inbounds %31[%417] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %416, %418 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %419 = llvm.add %417, %5 : i64
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
    %429 = llvm.add %427, %5 : i64
    llvm.store %429, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %430 = llvm.fpext %423 : f32 to f64
    %431 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %432 = llvm.getelementptr inbounds %31[%431] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %430, %432 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %433 = llvm.add %431, %5 : i64
    llvm.store %433, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %434 = llvm.fpext %421 : f32 to f64
    %435 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %436 = llvm.getelementptr inbounds %31[%435] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %434, %436 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %437 = llvm.add %435, %5 : i64
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
    %453 = llvm.add %451, %5 : i64
    llvm.store %453, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %454 = llvm.fpext %447 : f32 to f64
    %455 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %456 = llvm.getelementptr inbounds %31[%455] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %454, %456 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %457 = llvm.add %455, %5 : i64
    llvm.store %457, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %458 = llvm.fpext %445 : f32 to f64
    %459 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %460 = llvm.getelementptr inbounds %31[%459] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %458, %460 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %461 = llvm.add %459, %5 : i64
    llvm.store %461, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %462 = llvm.fpext %443 : f32 to f64
    %463 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %464 = llvm.getelementptr inbounds %31[%463] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %462, %464 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %465 = llvm.add %463, %5 : i64
    llvm.store %465, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %466 = llvm.fpext %441 : f32 to f64
    %467 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %468 = llvm.getelementptr inbounds %31[%467] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %466, %468 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %469 = llvm.add %467, %5 : i64
    llvm.store %469, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %470 = llvm.fpext %439 : f32 to f64
    %471 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %472 = llvm.getelementptr inbounds %31[%471] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %470, %472 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %473 = llvm.add %471, %5 : i64
    llvm.store %473, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %474 = llvm.fpext %43 : f32 to f64
    %475 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %476 = llvm.getelementptr inbounds %31[%475] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %474, %476 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %477 = llvm.add %475, %5 : i64
    llvm.store %477, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %478 = llvm.fpext %41 : f32 to f64
    %479 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %480 = llvm.getelementptr inbounds %31[%479] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %478, %480 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %481 = llvm.add %479, %5 : i64
    llvm.store %481, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %482 = llvm.fpext %39 : f32 to f64
    %483 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %484 = llvm.getelementptr inbounds %31[%483] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %482, %484 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %485 = llvm.add %483, %5 : i64
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
    %501 = llvm.add %499, %5 : i64
    llvm.store %501, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %502 = llvm.fpext %495 : f32 to f64
    %503 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %504 = llvm.getelementptr inbounds %31[%503] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %502, %504 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %505 = llvm.add %503, %5 : i64
    llvm.store %505, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %506 = llvm.fpext %493 : f32 to f64
    %507 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %508 = llvm.getelementptr inbounds %31[%507] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %506, %508 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %509 = llvm.add %507, %5 : i64
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
    %519 = llvm.add %517, %5 : i64
    llvm.store %519, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %520 = llvm.fpext %513 : f32 to f64
    %521 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %522 = llvm.getelementptr inbounds %31[%521] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %520, %522 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %523 = llvm.add %521, %5 : i64
    llvm.store %523, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %524 = llvm.fpext %511 : f32 to f64
    %525 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %526 = llvm.getelementptr inbounds %31[%525] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %524, %526 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %527 = llvm.add %525, %5 : i64
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
    %537 = llvm.add %535, %5 : i64
    llvm.store %537, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %538 = llvm.fpext %531 : f32 to f64
    %539 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %540 = llvm.getelementptr inbounds %31[%539] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %538, %540 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %541 = llvm.add %539, %5 : i64
    llvm.store %541, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %542 = llvm.fpext %529 : f32 to f64
    %543 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %544 = llvm.getelementptr inbounds %31[%543] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %542, %544 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %545 = llvm.add %543, %5 : i64
    llvm.store %545, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %546 = llvm.fpext %491 : f32 to f64
    %547 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %548 = llvm.getelementptr inbounds %31[%547] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %546, %548 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %549 = llvm.add %547, %5 : i64
    llvm.store %549, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %550 = llvm.fpext %489 : f32 to f64
    %551 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %552 = llvm.getelementptr inbounds %31[%551] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %550, %552 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %553 = llvm.add %551, %5 : i64
    llvm.store %553, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %554 = llvm.fpext %487 : f32 to f64
    %555 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %556 = llvm.getelementptr inbounds %31[%555] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %554, %556 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %557 = llvm.add %555, %5 : i64
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
    %573 = llvm.add %571, %5 : i64
    llvm.store %573, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %574 = llvm.fpext %567 : f32 to f64
    %575 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %576 = llvm.getelementptr inbounds %31[%575] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %574, %576 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %577 = llvm.add %575, %5 : i64
    llvm.store %577, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %578 = llvm.fpext %565 : f32 to f64
    %579 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %580 = llvm.getelementptr inbounds %31[%579] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %578, %580 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %581 = llvm.add %579, %5 : i64
    llvm.store %581, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %582 = llvm.fpext %563 : f32 to f64
    %583 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %584 = llvm.getelementptr inbounds %31[%583] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %582, %584 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %585 = llvm.add %583, %5 : i64
    llvm.store %585, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %586 = llvm.fpext %561 : f32 to f64
    %587 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %588 = llvm.getelementptr inbounds %31[%587] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %586, %588 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %589 = llvm.add %587, %5 : i64
    llvm.store %589, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %590 = llvm.fpext %559 : f32 to f64
    %591 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %592 = llvm.getelementptr inbounds %31[%591] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %590, %592 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %593 = llvm.add %591, %5 : i64
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
    %603 = llvm.add %601, %5 : i64
    llvm.store %603, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %604 = llvm.fpext %597 : f32 to f64
    %605 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %606 = llvm.getelementptr inbounds %31[%605] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %604, %606 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %607 = llvm.add %605, %5 : i64
    llvm.store %607, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %608 = llvm.fpext %595 : f32 to f64
    %609 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %610 = llvm.getelementptr inbounds %31[%609] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %608, %610 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %611 = llvm.add %609, %5 : i64
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
    %621 = llvm.add %619, %5 : i64
    llvm.store %621, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %622 = llvm.fpext %615 : f32 to f64
    %623 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %624 = llvm.getelementptr inbounds %31[%623] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %622, %624 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %625 = llvm.add %623, %5 : i64
    llvm.store %625, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %626 = llvm.fpext %613 : f32 to f64
    %627 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %628 = llvm.getelementptr inbounds %31[%627] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %626, %628 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %629 = llvm.add %627, %5 : i64
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
    %639 = llvm.add %637, %5 : i64
    llvm.store %639, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %640 = llvm.fpext %633 : f32 to f64
    %641 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %642 = llvm.getelementptr inbounds %31[%641] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %640, %642 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %643 = llvm.add %641, %5 : i64
    llvm.store %643, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %644 = llvm.fpext %631 : f32 to f64
    %645 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %646 = llvm.getelementptr inbounds %31[%645] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %644, %646 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %647 = llvm.add %645, %5 : i64
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
    %657 = llvm.add %655, %5 : i64
    llvm.store %657, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %658 = llvm.fpext %651 : f32 to f64
    %659 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %660 = llvm.getelementptr inbounds %31[%659] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %658, %660 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %661 = llvm.add %659, %5 : i64
    llvm.store %661, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %662 = llvm.fpext %649 : f32 to f64
    %663 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %664 = llvm.getelementptr inbounds %31[%663] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %662, %664 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %665 = llvm.add %663, %5 : i64
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
    %675 = llvm.add %673, %5 : i64
    llvm.store %675, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %676 = llvm.fpext %669 : f32 to f64
    %677 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %678 = llvm.getelementptr inbounds %31[%677] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %676, %678 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %679 = llvm.add %677, %5 : i64
    llvm.store %679, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    %680 = llvm.fpext %667 : f32 to f64
    %681 = llvm.load %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> i64
    %682 = llvm.getelementptr inbounds %31[%681] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %680, %682 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %683 = llvm.add %681, %5 : i64
    llvm.store %683, %37 {tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : i64, !llvm.ptr
    llvm.store %24, %28 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>, !llvm.ptr
    llvm.store %15, %27 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    llvm.store %36, %26 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %684 = llvm.getelementptr %3[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %685 = llvm.ptrtoint %684 : !llvm.ptr to i64
    %686 = llvm.call @_mlir_memref_to_llvm_alloc(%685) : (i64) -> !llvm.ptr
    %687 = llvm.insertvalue %686, %1[0] : !llvm.struct<(ptr, ptr, i64)> 
    %688 = llvm.insertvalue %686, %687[1] : !llvm.struct<(ptr, ptr, i64)> 
    %689 = llvm.insertvalue %6, %688[2] : !llvm.struct<(ptr, ptr, i64)> 
    llvm.store %689, %25 : !llvm.struct<(ptr, ptr, i64)>, !llvm.ptr
    llvm.call @qnode_forward_0.quantum(%28, %27, %26, %25) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.return %689 : !llvm.struct<(ptr, ptr, i64)>
  }
  llvm.func internal @_sample_loss.cloned(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: !llvm.ptr, %arg10: !llvm.ptr, %arg11: i64, %arg12: !llvm.ptr, %arg13: !llvm.ptr, %arg14: i64, %arg15: !llvm.ptr, %arg16: !llvm.ptr, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: !llvm.ptr, %arg21: !llvm.ptr, %arg22: i64, %arg23: !llvm.ptr, %arg24: !llvm.ptr, %arg25: i64, %arg26: !llvm.ptr, %arg27: !llvm.ptr, %arg28: i64) attributes {sym_visibility = "private"} {
    %0 = llvm.mlir.constant(64 : index) : i64
    %1 = llvm.mlir.zero : !llvm.ptr
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.mlir.constant(1.000000e+00 : f64) : f64
    %4 = llvm.mlir.constant(false) : i1
    %5 = llvm.mlir.constant(0.000000e+00 : f64) : f64
    %6 = llvm.call @qnode_forward_0.pcount(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg15, %arg16, %arg17, %arg18, %arg19) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> i64
    %7 = llvm.load %arg24 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %8 = llvm.load %arg21 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %9 = llvm.load %arg10 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %10 = llvm.load %arg13 {tbaa = [#llvm.tbaa_tag<base_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "float", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f32
    %11 = llvm.call @qnode_forward_0.preprocess(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg15, %arg16, %arg17, %arg18, %arg19, %6) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64) -> !llvm.struct<(ptr, ptr, i64)>
    %12 = llvm.extractvalue %11[1] : !llvm.struct<(ptr, ptr, i64)> 
    %13 = llvm.load %12 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    %14 = llvm.fpext %10 : f32 to f64
    %15 = llvm.fmul %14, %13 : f64
    %16 = llvm.fpext %9 : f32 to f64
    %17 = llvm.fadd %15, %16 : f64
    %18 = llvm.fpext %8 : f32 to f64
    %19 = llvm.fpext %7 : f32 to f64
    %20 = llvm.fcmp "ugt" %17, %5 : f64
    %21 = llvm.select %20, %17, %5 : i1, f64
    %22 = llvm.select %4, %5, %21 : i1, f64
    %23 = llvm.fcmp "une" %17, %17 : f64
    %24 = llvm.intr.fabs(%17) : (f64) -> f64
    %25 = llvm.fneg %24 : f64
    %26 = llvm.intr.exp(%25) : (f64) -> f64
    %27 = llvm.fadd %3, %26 : f64
    %28 = llvm.intr.log(%27) : (f64) -> f64
    %29 = llvm.fadd %22, %28 : f64
    %30 = llvm.select %23, %17, %29 : i1, f64
    %31 = llvm.fmul %18, %17 : f64
    %32 = llvm.fsub %30, %31 : f64
    %33 = llvm.fmul %19, %32 : f64
    %34 = llvm.getelementptr %1[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %35 = llvm.ptrtoint %34 : !llvm.ptr to i64
    %36 = llvm.add %35, %0 : i64
    %37 = llvm.call @_mlir_memref_to_llvm_alloc(%36) : (i64) -> !llvm.ptr
    %38 = llvm.ptrtoint %37 : !llvm.ptr to i64
    %39 = llvm.sub %0, %2 : i64
    %40 = llvm.add %38, %39 : i64
    %41 = llvm.urem %40, %0 : i64
    %42 = llvm.sub %40, %41 : i64
    %43 = llvm.inttoptr %42 : i64 to !llvm.ptr
    llvm.store %33, %43 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %44 = llvm.load %43 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.store %44, %arg27 {tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_root<id = "Catalyst TBAA">, 0>}>, offset = 0>]} : f64, !llvm.ptr
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
  llvm.func internal @softplus_2.detensorized(%arg0: f64) -> f64 attributes {sym_visibility = "private"} {
    %0 = llvm.mlir.constant(1.000000e+00 : f64) : f64
    %1 = llvm.mlir.constant(false) : i1
    %2 = llvm.mlir.constant(0.000000e+00 : f64) : f64
    %3 = llvm.fcmp "ugt" %arg0, %2 : f64
    %4 = llvm.select %3, %arg0, %2 : i1, f64
    %5 = llvm.select %1, %2, %4 : i1, f64
    %6 = llvm.fcmp "une" %arg0, %arg0 : f64
    %7 = llvm.intr.fabs(%arg0) : (f64) -> f64
    %8 = llvm.fneg %7 : f64
    %9 = llvm.intr.exp(%8) : (f64) -> f64
    %10 = llvm.fadd %0, %9 : f64
    %11 = llvm.intr.log(%10) : (f64) -> f64
    %12 = llvm.fadd %5, %11 : f64
    %13 = llvm.select %6, %arg0, %12 : i1, f64
    llvm.return %13 : f64
  }
}