module @qnode_forward {
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
  llvm.func @_mlir_memref_to_llvm_free(!llvm.ptr)
  llvm.func @_mlir_memref_to_llvm_alloc(i64) -> !llvm.ptr
  llvm.mlir.global private constant @__constant_xf32(3.14159274 : f32) {addr_space = 0 : i32, alignment = 64 : i64} : f32
  llvm.func @jit_qnode_forward(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: !llvm.ptr, %arg10: !llvm.ptr, %arg11: i64, %arg12: i64, %arg13: i64) -> !llvm.struct<(ptr, ptr, i64)> attributes {llvm.copy_memref, llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(0 : index) : i64
    %1 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64)>
    %2 = llvm.mlir.zero : !llvm.ptr
    %3 = llvm.mlir.constant(1 : index) : i64
    %4 = llvm.mlir.constant(3735928559 : index) : i64
    %5 = llvm.call @qnode_forward_0(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> !llvm.struct<(ptr, ptr, i64)>
    %6 = llvm.extractvalue %5[0] : !llvm.struct<(ptr, ptr, i64)> 
    %7 = llvm.ptrtoint %6 : !llvm.ptr to i64
    %8 = llvm.icmp "eq" %4, %7 : i64
    llvm.cond_br %8, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %9 = llvm.getelementptr %2[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %10 = llvm.ptrtoint %9 : !llvm.ptr to i64
    %11 = llvm.call @_mlir_memref_to_llvm_alloc(%10) : (i64) -> !llvm.ptr
    %12 = llvm.insertvalue %11, %1[0] : !llvm.struct<(ptr, ptr, i64)> 
    %13 = llvm.insertvalue %11, %12[1] : !llvm.struct<(ptr, ptr, i64)> 
    %14 = llvm.insertvalue %0, %13[2] : !llvm.struct<(ptr, ptr, i64)> 
    %15 = llvm.getelementptr %2[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %16 = llvm.ptrtoint %15 : !llvm.ptr to i64
    %17 = llvm.mul %16, %3 : i64
    %18 = llvm.extractvalue %5[1] : !llvm.struct<(ptr, ptr, i64)> 
    %19 = llvm.extractvalue %5[2] : !llvm.struct<(ptr, ptr, i64)> 
    %20 = llvm.getelementptr inbounds %18[%19] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    "llvm.intr.memcpy"(%11, %20, %17) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.br ^bb3(%14 : !llvm.struct<(ptr, ptr, i64)>)
  ^bb2:  // pred: ^bb0
    llvm.br ^bb3(%5 : !llvm.struct<(ptr, ptr, i64)>)
  ^bb3(%21: !llvm.struct<(ptr, ptr, i64)>):  // 2 preds: ^bb1, ^bb2
    llvm.br ^bb4
  ^bb4:  // pred: ^bb3
    llvm.return %21 : !llvm.struct<(ptr, ptr, i64)>
  }
  llvm.func @_catalyst_pyface_jit_qnode_forward(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
    %0 = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<(ptr, ptr, ptr)>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr, ptr, ptr)> 
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, ptr)> 
    llvm.call @_catalyst_ciface_jit_qnode_forward(%arg0, %1, %2) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }
  llvm.func @_catalyst_ciface_jit_qnode_forward(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) attributes {llvm.copy_memref, llvm.emit_c_interface} {
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
    %16 = llvm.call @jit_qnode_forward(%1, %2, %3, %4, %5, %6, %7, %8, %9, %11, %12, %13, %14, %15) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> !llvm.struct<(ptr, ptr, i64)>
    llvm.store %16, %arg0 : !llvm.struct<(ptr, ptr, i64)>, !llvm.ptr
    llvm.return
  }
  llvm.func internal @qnode_forward_0(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: !llvm.ptr, %arg10: !llvm.ptr, %arg11: i64, %arg12: i64, %arg13: i64) -> !llvm.struct<(ptr, ptr, i64)> attributes {diff_method = "adjoint", qnode} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.mlir.constant(2 : i64) : i64
    %2 = llvm.mlir.constant(3 : i64) : i64
    %3 = llvm.mlir.constant(4 : i64) : i64
    %4 = llvm.mlir.constant(false) : i1
    %5 = llvm.mlir.addressof @"{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}" : !llvm.ptr
    %6 = llvm.mlir.addressof @LightningSimulator : !llvm.ptr
    %7 = llvm.mlir.addressof @"/home/pichau/QAgents/.venv/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.so" : !llvm.ptr
    %8 = llvm.mlir.constant(64 : index) : i64
    %9 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64)>
    %10 = llvm.mlir.addressof @__constant_xf32 : !llvm.ptr
    %11 = llvm.mlir.zero : !llvm.ptr
    %12 = llvm.mlir.constant(0 : i64) : i64
    %13 = llvm.mlir.constant(0 : index) : i64
    %14 = llvm.mlir.constant(4 : index) : i64
    %15 = llvm.mlir.constant(1 : index) : i64
    %16 = llvm.getelementptr inbounds %arg1[14] : (!llvm.ptr) -> !llvm.ptr, f32
    %17 = llvm.load %16 : !llvm.ptr -> f32
    %18 = llvm.getelementptr inbounds %arg1[13] : (!llvm.ptr) -> !llvm.ptr, f32
    %19 = llvm.load %18 : !llvm.ptr -> f32
    %20 = llvm.getelementptr inbounds %arg1[12] : (!llvm.ptr) -> !llvm.ptr, f32
    %21 = llvm.load %20 : !llvm.ptr -> f32
    %22 = llvm.getelementptr inbounds %arg1[11] : (!llvm.ptr) -> !llvm.ptr, f32
    %23 = llvm.load %22 : !llvm.ptr -> f32
    %24 = llvm.getelementptr inbounds %arg1[10] : (!llvm.ptr) -> !llvm.ptr, f32
    %25 = llvm.load %24 : !llvm.ptr -> f32
    %26 = llvm.getelementptr inbounds %arg1[9] : (!llvm.ptr) -> !llvm.ptr, f32
    %27 = llvm.load %26 : !llvm.ptr -> f32
    %28 = llvm.getelementptr %11[4] : (!llvm.ptr) -> !llvm.ptr, f32
    %29 = llvm.ptrtoint %28 : !llvm.ptr to i64
    %30 = llvm.add %29, %8 : i64
    %31 = llvm.call @_mlir_memref_to_llvm_alloc(%30) : (i64) -> !llvm.ptr
    %32 = llvm.ptrtoint %31 : !llvm.ptr to i64
    %33 = llvm.sub %8, %15 : i64
    %34 = llvm.add %32, %33 : i64
    %35 = llvm.urem %34, %8 : i64
    %36 = llvm.sub %34, %35 : i64
    %37 = llvm.inttoptr %36 : i64 to !llvm.ptr
    llvm.br ^bb1(%13 : i64)
  ^bb1(%38: i64):  // 2 preds: ^bb0, ^bb2
    %39 = llvm.icmp "slt" %38, %14 : i64
    llvm.cond_br %39, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %40 = llvm.load %10 : !llvm.ptr -> f32
    %41 = llvm.getelementptr inbounds %37[%38] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %40, %41 : f32, !llvm.ptr
    %42 = llvm.add %38, %15 : i64
    llvm.br ^bb1(%42 : i64)
  ^bb3:  // pred: ^bb1
    llvm.br ^bb4(%13 : i64)
  ^bb4(%43: i64):  // 2 preds: ^bb3, ^bb5
    %44 = llvm.icmp "slt" %43, %14 : i64
    llvm.cond_br %44, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %45 = llvm.getelementptr inbounds %37[%43] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %46 = llvm.load %45 : !llvm.ptr -> f32
    %47 = llvm.getelementptr inbounds %arg10[%43] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %48 = llvm.load %47 : !llvm.ptr -> f32
    %49 = llvm.fmul %46, %48 : f32
    %50 = llvm.getelementptr inbounds %37[%43] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %49, %50 : f32, !llvm.ptr
    %51 = llvm.add %43, %15 : i64
    llvm.br ^bb4(%51 : i64)
  ^bb6:  // pred: ^bb4
    %52 = llvm.getelementptr inbounds %37[3] : (!llvm.ptr) -> !llvm.ptr, f32
    %53 = llvm.load %52 : !llvm.ptr -> f32
    %54 = llvm.getelementptr inbounds %7[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<107 x i8>
    %55 = llvm.getelementptr inbounds %6[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<19 x i8>
    %56 = llvm.getelementptr inbounds %5[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<54 x i8>
    llvm.call @__catalyst__rt__device_init(%54, %55, %56, %12, %4) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i1) -> ()
    %57 = llvm.call @__catalyst__rt__qubit_allocate_array(%3) : (i64) -> !llvm.ptr
    %58 = llvm.call @__catalyst__rt__array_get_element_ptr_1d(%57, %2) : (!llvm.ptr, i64) -> !llvm.ptr
    %59 = llvm.load %58 : !llvm.ptr -> !llvm.ptr
    %60 = llvm.fpext %53 : f32 to f64
    llvm.call @__catalyst__qis__RY(%60, %59, %11) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %61 = llvm.fpext %27 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%61, %59, %11) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %62 = llvm.fpext %25 : f32 to f64
    llvm.call @__catalyst__qis__RY(%62, %59, %11) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %63 = llvm.fpext %23 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%63, %59, %11) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %64 = llvm.getelementptr inbounds %arg1[8] : (!llvm.ptr) -> !llvm.ptr, f32
    %65 = llvm.load %64 : !llvm.ptr -> f32
    %66 = llvm.getelementptr inbounds %arg1[7] : (!llvm.ptr) -> !llvm.ptr, f32
    %67 = llvm.load %66 : !llvm.ptr -> f32
    %68 = llvm.getelementptr inbounds %arg1[6] : (!llvm.ptr) -> !llvm.ptr, f32
    %69 = llvm.load %68 : !llvm.ptr -> f32
    %70 = llvm.getelementptr inbounds %37[2] : (!llvm.ptr) -> !llvm.ptr, f32
    %71 = llvm.load %70 : !llvm.ptr -> f32
    %72 = llvm.call @__catalyst__rt__array_get_element_ptr_1d(%57, %1) : (!llvm.ptr, i64) -> !llvm.ptr
    %73 = llvm.load %72 : !llvm.ptr -> !llvm.ptr
    %74 = llvm.fpext %71 : f32 to f64
    llvm.call @__catalyst__qis__RY(%74, %73, %11) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %75 = llvm.fpext %69 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%75, %73, %11) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %76 = llvm.fpext %67 : f32 to f64
    llvm.call @__catalyst__qis__RY(%76, %73, %11) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %77 = llvm.fpext %65 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%77, %73, %11) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %78 = llvm.getelementptr inbounds %arg1[2] : (!llvm.ptr) -> !llvm.ptr, f32
    %79 = llvm.load %78 : !llvm.ptr -> f32
    %80 = llvm.getelementptr inbounds %arg1[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %81 = llvm.load %80 : !llvm.ptr -> f32
    %82 = llvm.load %arg1 : !llvm.ptr -> f32
    %83 = llvm.load %37 : !llvm.ptr -> f32
    %84 = llvm.call @__catalyst__rt__array_get_element_ptr_1d(%57, %12) : (!llvm.ptr, i64) -> !llvm.ptr
    %85 = llvm.load %84 : !llvm.ptr -> !llvm.ptr
    %86 = llvm.fpext %83 : f32 to f64
    llvm.call @__catalyst__qis__RY(%86, %85, %11) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %87 = llvm.fpext %82 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%87, %85, %11) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %88 = llvm.fpext %81 : f32 to f64
    llvm.call @__catalyst__qis__RY(%88, %85, %11) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %89 = llvm.fpext %79 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%89, %85, %11) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %90 = llvm.getelementptr inbounds %arg1[5] : (!llvm.ptr) -> !llvm.ptr, f32
    %91 = llvm.load %90 : !llvm.ptr -> f32
    %92 = llvm.getelementptr inbounds %arg1[4] : (!llvm.ptr) -> !llvm.ptr, f32
    %93 = llvm.load %92 : !llvm.ptr -> f32
    %94 = llvm.getelementptr inbounds %arg1[3] : (!llvm.ptr) -> !llvm.ptr, f32
    %95 = llvm.load %94 : !llvm.ptr -> f32
    %96 = llvm.getelementptr inbounds %37[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %97 = llvm.load %96 : !llvm.ptr -> f32
    llvm.call @_mlir_memref_to_llvm_free(%31) : (!llvm.ptr) -> ()
    %98 = llvm.call @__catalyst__rt__array_get_element_ptr_1d(%57, %0) : (!llvm.ptr, i64) -> !llvm.ptr
    %99 = llvm.load %98 : !llvm.ptr -> !llvm.ptr
    %100 = llvm.fpext %97 : f32 to f64
    llvm.call @__catalyst__qis__RY(%100, %99, %11) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %101 = llvm.fpext %95 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%101, %99, %11) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %102 = llvm.fpext %93 : f32 to f64
    llvm.call @__catalyst__qis__RY(%102, %99, %11) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %103 = llvm.fpext %91 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%103, %99, %11) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%85, %99, %11) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%99, %73, %11) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%73, %59, %11) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%59, %85, %11) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %104 = llvm.fpext %21 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%104, %85, %11) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %105 = llvm.fpext %19 : f32 to f64
    llvm.call @__catalyst__qis__RY(%105, %85, %11) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %106 = llvm.fpext %17 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%106, %85, %11) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %107 = llvm.getelementptr inbounds %arg1[20] : (!llvm.ptr) -> !llvm.ptr, f32
    %108 = llvm.load %107 : !llvm.ptr -> f32
    %109 = llvm.getelementptr inbounds %arg1[19] : (!llvm.ptr) -> !llvm.ptr, f32
    %110 = llvm.load %109 : !llvm.ptr -> f32
    %111 = llvm.getelementptr inbounds %arg1[18] : (!llvm.ptr) -> !llvm.ptr, f32
    %112 = llvm.load %111 : !llvm.ptr -> f32
    %113 = llvm.fpext %112 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%113, %73, %11) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %114 = llvm.fpext %110 : f32 to f64
    llvm.call @__catalyst__qis__RY(%114, %73, %11) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %115 = llvm.fpext %108 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%115, %73, %11) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%85, %73, %11) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%73, %85, %11) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %116 = llvm.getelementptr inbounds %arg1[17] : (!llvm.ptr) -> !llvm.ptr, f32
    %117 = llvm.load %116 : !llvm.ptr -> f32
    %118 = llvm.getelementptr inbounds %arg1[16] : (!llvm.ptr) -> !llvm.ptr, f32
    %119 = llvm.load %118 : !llvm.ptr -> f32
    %120 = llvm.getelementptr inbounds %arg1[15] : (!llvm.ptr) -> !llvm.ptr, f32
    %121 = llvm.load %120 : !llvm.ptr -> f32
    %122 = llvm.fpext %121 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%122, %99, %11) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %123 = llvm.fpext %119 : f32 to f64
    llvm.call @__catalyst__qis__RY(%123, %99, %11) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %124 = llvm.fpext %117 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%124, %99, %11) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %125 = llvm.getelementptr inbounds %arg1[23] : (!llvm.ptr) -> !llvm.ptr, f32
    %126 = llvm.load %125 : !llvm.ptr -> f32
    %127 = llvm.getelementptr inbounds %arg1[22] : (!llvm.ptr) -> !llvm.ptr, f32
    %128 = llvm.load %127 : !llvm.ptr -> f32
    %129 = llvm.getelementptr inbounds %arg1[21] : (!llvm.ptr) -> !llvm.ptr, f32
    %130 = llvm.load %129 : !llvm.ptr -> f32
    %131 = llvm.fpext %130 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%131, %59, %11) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %132 = llvm.fpext %128 : f32 to f64
    llvm.call @__catalyst__qis__RY(%132, %59, %11) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %133 = llvm.fpext %126 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%133, %59, %11) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%99, %59, %11) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%59, %99, %11) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %134 = llvm.call @__catalyst__qis__NamedObs(%2, %85) : (i64, !llvm.ptr) -> i64
    %135 = llvm.call @__catalyst__qis__Expval(%134) : (i64) -> f64
    %136 = llvm.getelementptr %11[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %137 = llvm.ptrtoint %136 : !llvm.ptr to i64
    %138 = llvm.add %137, %8 : i64
    %139 = llvm.call @_mlir_memref_to_llvm_alloc(%138) : (i64) -> !llvm.ptr
    %140 = llvm.ptrtoint %139 : !llvm.ptr to i64
    %141 = llvm.sub %8, %15 : i64
    %142 = llvm.add %140, %141 : i64
    %143 = llvm.urem %142, %8 : i64
    %144 = llvm.sub %142, %143 : i64
    %145 = llvm.inttoptr %144 : i64 to !llvm.ptr
    %146 = llvm.insertvalue %139, %9[0] : !llvm.struct<(ptr, ptr, i64)> 
    %147 = llvm.insertvalue %145, %146[1] : !llvm.struct<(ptr, ptr, i64)> 
    %148 = llvm.insertvalue %13, %147[2] : !llvm.struct<(ptr, ptr, i64)> 
    llvm.store %135, %145 : f64, !llvm.ptr
    llvm.call @__catalyst__rt__qubit_release_array(%57) : (!llvm.ptr) -> ()
    llvm.call @__catalyst__rt__device_release() : () -> ()
    llvm.return %148 : !llvm.struct<(ptr, ptr, i64)>
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