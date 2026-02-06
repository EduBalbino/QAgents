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
  llvm.mlir.global internal constant @"{}"("{}\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @LightningGPUSimulator("LightningGPUSimulator\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @"/home/pichau/QAgents/.venv/lib/python3.12/site-packages/pennylane_lightning/liblightning_gpu_catalyst.so"("/home/pichau/QAgents/.venv/lib/python3.12/site-packages/pennylane_lightning/liblightning_gpu_catalyst.so\00") {addr_space = 0 : i32}
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
    %21 = llvm.load %20 : !llvm.ptr -> f32
    %22 = llvm.getelementptr inbounds %arg1[73] : (!llvm.ptr) -> !llvm.ptr, f32
    %23 = llvm.load %22 : !llvm.ptr -> f32
    %24 = llvm.getelementptr inbounds %arg1[72] : (!llvm.ptr) -> !llvm.ptr, f32
    %25 = llvm.load %24 : !llvm.ptr -> f32
    %26 = llvm.getelementptr inbounds %arg1[50] : (!llvm.ptr) -> !llvm.ptr, f32
    %27 = llvm.load %26 : !llvm.ptr -> f32
    %28 = llvm.getelementptr inbounds %arg1[49] : (!llvm.ptr) -> !llvm.ptr, f32
    %29 = llvm.load %28 : !llvm.ptr -> f32
    %30 = llvm.getelementptr inbounds %arg1[48] : (!llvm.ptr) -> !llvm.ptr, f32
    %31 = llvm.load %30 : !llvm.ptr -> f32
    %32 = llvm.getelementptr inbounds %arg1[44] : (!llvm.ptr) -> !llvm.ptr, f32
    %33 = llvm.load %32 : !llvm.ptr -> f32
    %34 = llvm.getelementptr inbounds %arg1[43] : (!llvm.ptr) -> !llvm.ptr, f32
    %35 = llvm.load %34 : !llvm.ptr -> f32
    %36 = llvm.getelementptr inbounds %arg1[42] : (!llvm.ptr) -> !llvm.ptr, f32
    %37 = llvm.load %36 : !llvm.ptr -> f32
    %38 = llvm.getelementptr inbounds %arg1[23] : (!llvm.ptr) -> !llvm.ptr, f32
    %39 = llvm.load %38 : !llvm.ptr -> f32
    %40 = llvm.getelementptr inbounds %arg1[22] : (!llvm.ptr) -> !llvm.ptr, f32
    %41 = llvm.load %40 : !llvm.ptr -> f32
    %42 = llvm.getelementptr inbounds %arg1[21] : (!llvm.ptr) -> !llvm.ptr, f32
    %43 = llvm.load %42 : !llvm.ptr -> f32
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
    %56 = llvm.load %14 : !llvm.ptr -> f32
    %57 = llvm.getelementptr inbounds %53[%54] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %56, %57 : f32, !llvm.ptr
    %58 = llvm.add %54, %19 : i64
    llvm.br ^bb1(%58 : i64)
  ^bb3:  // pred: ^bb1
    llvm.br ^bb4(%17 : i64)
  ^bb4(%59: i64):  // 2 preds: ^bb3, ^bb5
    %60 = llvm.icmp "slt" %59, %18 : i64
    llvm.cond_br %60, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %61 = llvm.getelementptr inbounds %53[%59] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %62 = llvm.load %61 : !llvm.ptr -> f32
    %63 = llvm.getelementptr inbounds %arg10[%59] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %64 = llvm.load %63 : !llvm.ptr -> f32
    %65 = llvm.fmul %62, %64 : f32
    %66 = llvm.getelementptr inbounds %53[%59] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %65, %66 : f32, !llvm.ptr
    %67 = llvm.add %59, %19 : i64
    llvm.br ^bb4(%67 : i64)
  ^bb6:  // pred: ^bb4
    %68 = llvm.getelementptr inbounds %53[7] : (!llvm.ptr) -> !llvm.ptr, f32
    %69 = llvm.load %68 : !llvm.ptr -> f32
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
    %81 = llvm.load %80 : !llvm.ptr -> f32
    %82 = llvm.getelementptr inbounds %arg1[19] : (!llvm.ptr) -> !llvm.ptr, f32
    %83 = llvm.load %82 : !llvm.ptr -> f32
    %84 = llvm.getelementptr inbounds %arg1[18] : (!llvm.ptr) -> !llvm.ptr, f32
    %85 = llvm.load %84 : !llvm.ptr -> f32
    %86 = llvm.getelementptr inbounds %53[6] : (!llvm.ptr) -> !llvm.ptr, f32
    %87 = llvm.load %86 : !llvm.ptr -> f32
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
    %95 = llvm.load %94 : !llvm.ptr -> f32
    %96 = llvm.getelementptr inbounds %arg1[16] : (!llvm.ptr) -> !llvm.ptr, f32
    %97 = llvm.load %96 : !llvm.ptr -> f32
    %98 = llvm.getelementptr inbounds %arg1[15] : (!llvm.ptr) -> !llvm.ptr, f32
    %99 = llvm.load %98 : !llvm.ptr -> f32
    %100 = llvm.getelementptr inbounds %53[5] : (!llvm.ptr) -> !llvm.ptr, f32
    %101 = llvm.load %100 : !llvm.ptr -> f32
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
    %109 = llvm.load %108 : !llvm.ptr -> f32
    %110 = llvm.getelementptr inbounds %arg1[13] : (!llvm.ptr) -> !llvm.ptr, f32
    %111 = llvm.load %110 : !llvm.ptr -> f32
    %112 = llvm.getelementptr inbounds %arg1[12] : (!llvm.ptr) -> !llvm.ptr, f32
    %113 = llvm.load %112 : !llvm.ptr -> f32
    %114 = llvm.getelementptr inbounds %53[4] : (!llvm.ptr) -> !llvm.ptr, f32
    %115 = llvm.load %114 : !llvm.ptr -> f32
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
    %123 = llvm.load %122 : !llvm.ptr -> f32
    %124 = llvm.getelementptr inbounds %arg1[10] : (!llvm.ptr) -> !llvm.ptr, f32
    %125 = llvm.load %124 : !llvm.ptr -> f32
    %126 = llvm.getelementptr inbounds %arg1[9] : (!llvm.ptr) -> !llvm.ptr, f32
    %127 = llvm.load %126 : !llvm.ptr -> f32
    %128 = llvm.getelementptr inbounds %53[3] : (!llvm.ptr) -> !llvm.ptr, f32
    %129 = llvm.load %128 : !llvm.ptr -> f32
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
    %137 = llvm.load %136 : !llvm.ptr -> f32
    %138 = llvm.getelementptr inbounds %arg1[7] : (!llvm.ptr) -> !llvm.ptr, f32
    %139 = llvm.load %138 : !llvm.ptr -> f32
    %140 = llvm.getelementptr inbounds %arg1[6] : (!llvm.ptr) -> !llvm.ptr, f32
    %141 = llvm.load %140 : !llvm.ptr -> f32
    %142 = llvm.getelementptr inbounds %53[2] : (!llvm.ptr) -> !llvm.ptr, f32
    %143 = llvm.load %142 : !llvm.ptr -> f32
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
    %151 = llvm.load %150 : !llvm.ptr -> f32
    %152 = llvm.getelementptr inbounds %arg1[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %153 = llvm.load %152 : !llvm.ptr -> f32
    %154 = llvm.load %arg1 : !llvm.ptr -> f32
    %155 = llvm.load %53 : !llvm.ptr -> f32
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
    %163 = llvm.load %162 : !llvm.ptr -> f32
    %164 = llvm.getelementptr inbounds %arg1[4] : (!llvm.ptr) -> !llvm.ptr, f32
    %165 = llvm.load %164 : !llvm.ptr -> f32
    %166 = llvm.getelementptr inbounds %arg1[3] : (!llvm.ptr) -> !llvm.ptr, f32
    %167 = llvm.load %166 : !llvm.ptr -> f32
    %168 = llvm.getelementptr inbounds %53[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %169 = llvm.load %168 : !llvm.ptr -> f32
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
    %180 = llvm.load %179 : !llvm.ptr -> f32
    %181 = llvm.getelementptr inbounds %arg1[37] : (!llvm.ptr) -> !llvm.ptr, f32
    %182 = llvm.load %181 : !llvm.ptr -> f32
    %183 = llvm.getelementptr inbounds %arg1[36] : (!llvm.ptr) -> !llvm.ptr, f32
    %184 = llvm.load %183 : !llvm.ptr -> f32
    %185 = llvm.fpext %184 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%185, %117, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %186 = llvm.fpext %182 : f32 to f64
    llvm.call @__catalyst__qis__RY(%186, %117, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %187 = llvm.fpext %180 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%187, %117, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %188 = llvm.getelementptr inbounds %arg1[26] : (!llvm.ptr) -> !llvm.ptr, f32
    %189 = llvm.load %188 : !llvm.ptr -> f32
    %190 = llvm.getelementptr inbounds %arg1[25] : (!llvm.ptr) -> !llvm.ptr, f32
    %191 = llvm.load %190 : !llvm.ptr -> f32
    %192 = llvm.getelementptr inbounds %arg1[24] : (!llvm.ptr) -> !llvm.ptr, f32
    %193 = llvm.load %192 : !llvm.ptr -> f32
    llvm.call @__catalyst__qis__CNOT(%75, %157, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %194 = llvm.fpext %193 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%194, %157, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %195 = llvm.fpext %191 : f32 to f64
    llvm.call @__catalyst__qis__RY(%195, %157, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %196 = llvm.fpext %189 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%196, %157, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %197 = llvm.getelementptr inbounds %arg1[32] : (!llvm.ptr) -> !llvm.ptr, f32
    %198 = llvm.load %197 : !llvm.ptr -> f32
    %199 = llvm.getelementptr inbounds %arg1[31] : (!llvm.ptr) -> !llvm.ptr, f32
    %200 = llvm.load %199 : !llvm.ptr -> f32
    %201 = llvm.getelementptr inbounds %arg1[30] : (!llvm.ptr) -> !llvm.ptr, f32
    %202 = llvm.load %201 : !llvm.ptr -> f32
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
    %210 = llvm.load %209 : !llvm.ptr -> f32
    %211 = llvm.getelementptr inbounds %arg1[58] : (!llvm.ptr) -> !llvm.ptr, f32
    %212 = llvm.load %211 : !llvm.ptr -> f32
    %213 = llvm.getelementptr inbounds %arg1[57] : (!llvm.ptr) -> !llvm.ptr, f32
    %214 = llvm.load %213 : !llvm.ptr -> f32
    %215 = llvm.getelementptr inbounds %arg1[41] : (!llvm.ptr) -> !llvm.ptr, f32
    %216 = llvm.load %215 : !llvm.ptr -> f32
    %217 = llvm.getelementptr inbounds %arg1[40] : (!llvm.ptr) -> !llvm.ptr, f32
    %218 = llvm.load %217 : !llvm.ptr -> f32
    %219 = llvm.getelementptr inbounds %arg1[39] : (!llvm.ptr) -> !llvm.ptr, f32
    %220 = llvm.load %219 : !llvm.ptr -> f32
    %221 = llvm.fpext %220 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%221, %103, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %222 = llvm.fpext %218 : f32 to f64
    llvm.call @__catalyst__qis__RY(%222, %103, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %223 = llvm.fpext %216 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%223, %103, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %224 = llvm.getelementptr inbounds %arg1[29] : (!llvm.ptr) -> !llvm.ptr, f32
    %225 = llvm.load %224 : !llvm.ptr -> f32
    %226 = llvm.getelementptr inbounds %arg1[28] : (!llvm.ptr) -> !llvm.ptr, f32
    %227 = llvm.load %226 : !llvm.ptr -> f32
    %228 = llvm.getelementptr inbounds %arg1[27] : (!llvm.ptr) -> !llvm.ptr, f32
    %229 = llvm.load %228 : !llvm.ptr -> f32
    %230 = llvm.fpext %229 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%230, %171, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %231 = llvm.fpext %227 : f32 to f64
    llvm.call @__catalyst__qis__RY(%231, %171, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %232 = llvm.fpext %225 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%232, %171, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %233 = llvm.getelementptr inbounds %arg1[35] : (!llvm.ptr) -> !llvm.ptr, f32
    %234 = llvm.load %233 : !llvm.ptr -> f32
    %235 = llvm.getelementptr inbounds %arg1[34] : (!llvm.ptr) -> !llvm.ptr, f32
    %236 = llvm.load %235 : !llvm.ptr -> f32
    %237 = llvm.getelementptr inbounds %arg1[33] : (!llvm.ptr) -> !llvm.ptr, f32
    %238 = llvm.load %237 : !llvm.ptr -> f32
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
    %246 = llvm.load %245 : !llvm.ptr -> f32
    %247 = llvm.getelementptr inbounds %arg1[55] : (!llvm.ptr) -> !llvm.ptr, f32
    %248 = llvm.load %247 : !llvm.ptr -> f32
    %249 = llvm.getelementptr inbounds %arg1[54] : (!llvm.ptr) -> !llvm.ptr, f32
    %250 = llvm.load %249 : !llvm.ptr -> f32
    %251 = llvm.fpext %250 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%251, %145, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %252 = llvm.fpext %248 : f32 to f64
    llvm.call @__catalyst__qis__RY(%252, %145, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %253 = llvm.fpext %246 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%253, %145, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %254 = llvm.getelementptr inbounds %arg1[65] : (!llvm.ptr) -> !llvm.ptr, f32
    %255 = llvm.load %254 : !llvm.ptr -> f32
    %256 = llvm.getelementptr inbounds %arg1[64] : (!llvm.ptr) -> !llvm.ptr, f32
    %257 = llvm.load %256 : !llvm.ptr -> f32
    %258 = llvm.getelementptr inbounds %arg1[63] : (!llvm.ptr) -> !llvm.ptr, f32
    %259 = llvm.load %258 : !llvm.ptr -> f32
    %260 = llvm.getelementptr inbounds %arg1[47] : (!llvm.ptr) -> !llvm.ptr, f32
    %261 = llvm.load %260 : !llvm.ptr -> f32
    %262 = llvm.getelementptr inbounds %arg1[46] : (!llvm.ptr) -> !llvm.ptr, f32
    %263 = llvm.load %262 : !llvm.ptr -> f32
    %264 = llvm.getelementptr inbounds %arg1[45] : (!llvm.ptr) -> !llvm.ptr, f32
    %265 = llvm.load %264 : !llvm.ptr -> f32
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
    %276 = llvm.load %275 : !llvm.ptr -> f32
    %277 = llvm.getelementptr inbounds %arg1[85] : (!llvm.ptr) -> !llvm.ptr, f32
    %278 = llvm.load %277 : !llvm.ptr -> f32
    %279 = llvm.getelementptr inbounds %arg1[84] : (!llvm.ptr) -> !llvm.ptr, f32
    %280 = llvm.load %279 : !llvm.ptr -> f32
    %281 = llvm.getelementptr inbounds %arg1[71] : (!llvm.ptr) -> !llvm.ptr, f32
    %282 = llvm.load %281 : !llvm.ptr -> f32
    %283 = llvm.getelementptr inbounds %arg1[70] : (!llvm.ptr) -> !llvm.ptr, f32
    %284 = llvm.load %283 : !llvm.ptr -> f32
    %285 = llvm.getelementptr inbounds %arg1[69] : (!llvm.ptr) -> !llvm.ptr, f32
    %286 = llvm.load %285 : !llvm.ptr -> f32
    llvm.call @__catalyst__qis__CNOT(%75, %171, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %287 = llvm.fpext %286 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%287, %75, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %288 = llvm.fpext %284 : f32 to f64
    llvm.call @__catalyst__qis__RY(%288, %75, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %289 = llvm.fpext %282 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%289, %75, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %290 = llvm.getelementptr inbounds %arg1[53] : (!llvm.ptr) -> !llvm.ptr, f32
    %291 = llvm.load %290 : !llvm.ptr -> f32
    %292 = llvm.getelementptr inbounds %arg1[52] : (!llvm.ptr) -> !llvm.ptr, f32
    %293 = llvm.load %292 : !llvm.ptr -> f32
    %294 = llvm.getelementptr inbounds %arg1[51] : (!llvm.ptr) -> !llvm.ptr, f32
    %295 = llvm.load %294 : !llvm.ptr -> f32
    %296 = llvm.fpext %295 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%296, %171, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %297 = llvm.fpext %293 : f32 to f64
    llvm.call @__catalyst__qis__RY(%297, %171, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %298 = llvm.fpext %291 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%298, %171, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %299 = llvm.getelementptr inbounds %arg1[62] : (!llvm.ptr) -> !llvm.ptr, f32
    %300 = llvm.load %299 : !llvm.ptr -> f32
    %301 = llvm.getelementptr inbounds %arg1[61] : (!llvm.ptr) -> !llvm.ptr, f32
    %302 = llvm.load %301 : !llvm.ptr -> f32
    %303 = llvm.getelementptr inbounds %arg1[60] : (!llvm.ptr) -> !llvm.ptr, f32
    %304 = llvm.load %303 : !llvm.ptr -> f32
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
    %312 = llvm.load %311 : !llvm.ptr -> f32
    %313 = llvm.getelementptr inbounds %arg1[76] : (!llvm.ptr) -> !llvm.ptr, f32
    %314 = llvm.load %313 : !llvm.ptr -> f32
    %315 = llvm.getelementptr inbounds %arg1[75] : (!llvm.ptr) -> !llvm.ptr, f32
    %316 = llvm.load %315 : !llvm.ptr -> f32
    %317 = llvm.getelementptr inbounds %arg1[68] : (!llvm.ptr) -> !llvm.ptr, f32
    %318 = llvm.load %317 : !llvm.ptr -> f32
    %319 = llvm.getelementptr inbounds %arg1[67] : (!llvm.ptr) -> !llvm.ptr, f32
    %320 = llvm.load %319 : !llvm.ptr -> f32
    %321 = llvm.getelementptr inbounds %arg1[66] : (!llvm.ptr) -> !llvm.ptr, f32
    %322 = llvm.load %321 : !llvm.ptr -> f32
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
    %330 = llvm.load %329 : !llvm.ptr -> f32
    %331 = llvm.getelementptr inbounds %arg1[88] : (!llvm.ptr) -> !llvm.ptr, f32
    %332 = llvm.load %331 : !llvm.ptr -> f32
    %333 = llvm.getelementptr inbounds %arg1[87] : (!llvm.ptr) -> !llvm.ptr, f32
    %334 = llvm.load %333 : !llvm.ptr -> f32
    %335 = llvm.fpext %334 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%335, %103, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %336 = llvm.fpext %332 : f32 to f64
    llvm.call @__catalyst__qis__RY(%336, %103, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %337 = llvm.fpext %330 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%337, %103, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%171, %103, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%103, %171, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %338 = llvm.getelementptr inbounds %arg1[80] : (!llvm.ptr) -> !llvm.ptr, f32
    %339 = llvm.load %338 : !llvm.ptr -> f32
    %340 = llvm.getelementptr inbounds %arg1[79] : (!llvm.ptr) -> !llvm.ptr, f32
    %341 = llvm.load %340 : !llvm.ptr -> f32
    %342 = llvm.getelementptr inbounds %arg1[78] : (!llvm.ptr) -> !llvm.ptr, f32
    %343 = llvm.load %342 : !llvm.ptr -> f32
    llvm.call @__catalyst__qis__CNOT(%75, %145, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %344 = llvm.fpext %343 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%344, %145, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %345 = llvm.fpext %341 : f32 to f64
    llvm.call @__catalyst__qis__RY(%345, %145, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %346 = llvm.fpext %339 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%346, %145, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %347 = llvm.getelementptr inbounds %arg1[92] : (!llvm.ptr) -> !llvm.ptr, f32
    %348 = llvm.load %347 : !llvm.ptr -> f32
    %349 = llvm.getelementptr inbounds %arg1[91] : (!llvm.ptr) -> !llvm.ptr, f32
    %350 = llvm.load %349 : !llvm.ptr -> f32
    %351 = llvm.getelementptr inbounds %arg1[90] : (!llvm.ptr) -> !llvm.ptr, f32
    %352 = llvm.load %351 : !llvm.ptr -> f32
    %353 = llvm.fpext %352 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%353, %89, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %354 = llvm.fpext %350 : f32 to f64
    llvm.call @__catalyst__qis__RY(%354, %89, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %355 = llvm.fpext %348 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%355, %89, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%145, %89, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%89, %145, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %356 = llvm.getelementptr inbounds %arg1[83] : (!llvm.ptr) -> !llvm.ptr, f32
    %357 = llvm.load %356 : !llvm.ptr -> f32
    %358 = llvm.getelementptr inbounds %arg1[82] : (!llvm.ptr) -> !llvm.ptr, f32
    %359 = llvm.load %358 : !llvm.ptr -> f32
    %360 = llvm.getelementptr inbounds %arg1[81] : (!llvm.ptr) -> !llvm.ptr, f32
    %361 = llvm.load %360 : !llvm.ptr -> f32
    %362 = llvm.fpext %361 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%362, %131, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %363 = llvm.fpext %359 : f32 to f64
    llvm.call @__catalyst__qis__RY(%363, %131, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %364 = llvm.fpext %357 : f32 to f64
    llvm.call @__catalyst__qis__RZ(%364, %131, %15) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %365 = llvm.getelementptr inbounds %arg1[95] : (!llvm.ptr) -> !llvm.ptr, f32
    %366 = llvm.load %365 : !llvm.ptr -> f32
    %367 = llvm.getelementptr inbounds %arg1[94] : (!llvm.ptr) -> !llvm.ptr, f32
    %368 = llvm.load %367 : !llvm.ptr -> f32
    %369 = llvm.getelementptr inbounds %arg1[93] : (!llvm.ptr) -> !llvm.ptr, f32
    %370 = llvm.load %369 : !llvm.ptr -> f32
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
    llvm.store %375, %385 : f64, !llvm.ptr
    llvm.call @__catalyst__rt__qubit_release_array(%73) : (!llvm.ptr) -> ()
    llvm.call @__catalyst__rt__device_release() : () -> ()
    llvm.return %388 : !llvm.struct<(ptr, ptr, i64)>
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