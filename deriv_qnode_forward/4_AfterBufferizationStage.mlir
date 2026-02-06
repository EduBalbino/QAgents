module @deriv_qnode_forward {
  memref.global "private" constant @__constant_xf32 : memref<f32> = dense<3.14159274> {alignment = 64 : i64}
  func.func public @jit_deriv_qnode_forward(%arg0: memref<4x8x3xf32>, %arg1: memref<8xf32>) -> memref<4x8x3xf32> attributes {llvm.copy_memref, llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(3735928559 : index) : i64
    %cst = arith.constant 1.000000e+00 : f64
    %cst_0 = arith.constant 0.000000e+00 : f64
    %1 = call @qnode_forward_0.pcount(%arg0, %arg1) : (memref<4x8x3xf32>, memref<8xf32>) -> index
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<f64>
    linalg.fill ins(%cst_0 : f64) outs(%alloc : memref<f64>)
    memref.store %cst, %alloc[] : memref<f64>
    %alloc_1 = memref.alloc() : memref<4x8x3xf32>
    %alloc_2 = memref.alloc() : memref<f64>
    gradient.backprop @qnode_forward_0.preprocess(%arg0, %arg1, %1) grad_out(%alloc_1 : memref<4x8x3xf32>) callee_out(%alloc_2 : memref<f64>) cotangents(%alloc : memref<f64>) {diffArgIndices = dense<0> : tensor<1xi64>, keepValueResults = false, resultSegmentSizes = array<i32: 0, 0>} : (memref<4x8x3xf32>, memref<8xf32>, index) -> ()
    memref.dealloc %alloc_2 : memref<f64>
    memref.dealloc %alloc : memref<f64>
    %2 = builtin.unrealized_conversion_cast %alloc_1 : memref<4x8x3xf32> to !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %3 = llvm.extractvalue %2[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %4 = llvm.ptrtoint %3 : !llvm.ptr to i64
    %5 = llvm.icmp "eq" %0, %4 : i64
    %6 = scf.if %5 -> (memref<4x8x3xf32>) {
      %alloc_3 = memref.alloc() : memref<4x8x3xf32>
      memref.copy %alloc_1, %alloc_3 : memref<4x8x3xf32> to memref<4x8x3xf32>
      scf.yield %alloc_3 : memref<4x8x3xf32>
    } else {
      scf.yield %alloc_1 : memref<4x8x3xf32>
    }
    return %6 : memref<4x8x3xf32>
  }
  func.func public @qnode_forward_0(%arg0: memref<4x8x3xf32>, %arg1: memref<8xf32>) -> memref<f64> attributes {diff_method = "adjoint", llvm.linkage = #llvm.linkage<internal>, qnode} {
    %c0_i64 = arith.constant 0 : i64
    %0 = memref.get_global @__constant_xf32 : memref<f32>
    %subview = memref.subview %arg0[3, 0, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 74>>
    %collapse_shape = memref.collapse_shape %subview [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 74>> into memref<f32, strided<[], offset: 74>>
    %1 = memref.load %collapse_shape[] : memref<f32, strided<[], offset: 74>>
    %subview_0 = memref.subview %arg0[3, 0, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 73>>
    %collapse_shape_1 = memref.collapse_shape %subview_0 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 73>> into memref<f32, strided<[], offset: 73>>
    %2 = memref.load %collapse_shape_1[] : memref<f32, strided<[], offset: 73>>
    %subview_2 = memref.subview %arg0[3, 0, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 72>>
    %collapse_shape_3 = memref.collapse_shape %subview_2 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 72>> into memref<f32, strided<[], offset: 72>>
    %3 = memref.load %collapse_shape_3[] : memref<f32, strided<[], offset: 72>>
    %subview_4 = memref.subview %arg0[2, 0, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 50>>
    %collapse_shape_5 = memref.collapse_shape %subview_4 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 50>> into memref<f32, strided<[], offset: 50>>
    %4 = memref.load %collapse_shape_5[] : memref<f32, strided<[], offset: 50>>
    %subview_6 = memref.subview %arg0[2, 0, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 49>>
    %collapse_shape_7 = memref.collapse_shape %subview_6 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 49>> into memref<f32, strided<[], offset: 49>>
    %5 = memref.load %collapse_shape_7[] : memref<f32, strided<[], offset: 49>>
    %subview_8 = memref.subview %arg0[2, 0, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 48>>
    %collapse_shape_9 = memref.collapse_shape %subview_8 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 48>> into memref<f32, strided<[], offset: 48>>
    %6 = memref.load %collapse_shape_9[] : memref<f32, strided<[], offset: 48>>
    %subview_10 = memref.subview %arg0[1, 6, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 44>>
    %collapse_shape_11 = memref.collapse_shape %subview_10 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 44>> into memref<f32, strided<[], offset: 44>>
    %7 = memref.load %collapse_shape_11[] : memref<f32, strided<[], offset: 44>>
    %subview_12 = memref.subview %arg0[1, 6, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 43>>
    %collapse_shape_13 = memref.collapse_shape %subview_12 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 43>> into memref<f32, strided<[], offset: 43>>
    %8 = memref.load %collapse_shape_13[] : memref<f32, strided<[], offset: 43>>
    %subview_14 = memref.subview %arg0[1, 6, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 42>>
    %collapse_shape_15 = memref.collapse_shape %subview_14 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 42>> into memref<f32, strided<[], offset: 42>>
    %9 = memref.load %collapse_shape_15[] : memref<f32, strided<[], offset: 42>>
    %subview_16 = memref.subview %arg0[0, 7, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 23>>
    %collapse_shape_17 = memref.collapse_shape %subview_16 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 23>> into memref<f32, strided<[], offset: 23>>
    %10 = memref.load %collapse_shape_17[] : memref<f32, strided<[], offset: 23>>
    %subview_18 = memref.subview %arg0[0, 7, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 22>>
    %collapse_shape_19 = memref.collapse_shape %subview_18 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 22>> into memref<f32, strided<[], offset: 22>>
    %11 = memref.load %collapse_shape_19[] : memref<f32, strided<[], offset: 22>>
    %subview_20 = memref.subview %arg0[0, 7, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 21>>
    %collapse_shape_21 = memref.collapse_shape %subview_20 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 21>> into memref<f32, strided<[], offset: 21>>
    %12 = memref.load %collapse_shape_21[] : memref<f32, strided<[], offset: 21>>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<8xf32>
    linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%0 : memref<f32>) outs(%alloc : memref<8xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%alloc, %arg1 : memref<8xf32>, memref<8xf32>) outs(%alloc : memref<8xf32>) {
    ^bb0(%in: f32, %in_342: f32, %out: f32):
      %228 = arith.mulf %in, %in_342 : f32
      linalg.yield %228 : f32
    }
    %subview_22 = memref.subview %alloc[7] [1] [1] : memref<8xf32> to memref<1xf32, strided<[1], offset: 7>>
    %collapse_shape_23 = memref.collapse_shape %subview_22 [] : memref<1xf32, strided<[1], offset: 7>> into memref<f32, strided<[], offset: 7>>
    %13 = memref.load %collapse_shape_23[] : memref<f32, strided<[], offset: 7>>
    quantum.device shots(%c0_i64) ["/home/pichau/QAgents/.venv/lib/python3.12/site-packages/pennylane_lightning/liblightning_gpu_catalyst.so", "LightningGPUSimulator", "{}"]
    %14 = quantum.alloc( 8) : !quantum.reg
    %15 = quantum.extract %14[ 7] : !quantum.reg -> !quantum.bit
    %16 = arith.extf %13 : f32 to f64
    %out_qubits = quantum.custom "RY"(%16) %15 : !quantum.bit
    %17 = arith.extf %12 : f32 to f64
    %out_qubits_24 = quantum.custom "RZ"(%17) %out_qubits : !quantum.bit
    %18 = arith.extf %11 : f32 to f64
    %out_qubits_25 = quantum.custom "RY"(%18) %out_qubits_24 : !quantum.bit
    %19 = arith.extf %10 : f32 to f64
    %out_qubits_26 = quantum.custom "RZ"(%19) %out_qubits_25 : !quantum.bit
    %subview_27 = memref.subview %arg0[0, 6, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 20>>
    %collapse_shape_28 = memref.collapse_shape %subview_27 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 20>> into memref<f32, strided<[], offset: 20>>
    %20 = memref.load %collapse_shape_28[] : memref<f32, strided<[], offset: 20>>
    %subview_29 = memref.subview %arg0[0, 6, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 19>>
    %collapse_shape_30 = memref.collapse_shape %subview_29 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 19>> into memref<f32, strided<[], offset: 19>>
    %21 = memref.load %collapse_shape_30[] : memref<f32, strided<[], offset: 19>>
    %subview_31 = memref.subview %arg0[0, 6, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 18>>
    %collapse_shape_32 = memref.collapse_shape %subview_31 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 18>> into memref<f32, strided<[], offset: 18>>
    %22 = memref.load %collapse_shape_32[] : memref<f32, strided<[], offset: 18>>
    %subview_33 = memref.subview %alloc[6] [1] [1] : memref<8xf32> to memref<1xf32, strided<[1], offset: 6>>
    %collapse_shape_34 = memref.collapse_shape %subview_33 [] : memref<1xf32, strided<[1], offset: 6>> into memref<f32, strided<[], offset: 6>>
    %23 = memref.load %collapse_shape_34[] : memref<f32, strided<[], offset: 6>>
    %24 = quantum.extract %14[ 6] : !quantum.reg -> !quantum.bit
    %25 = arith.extf %23 : f32 to f64
    %out_qubits_35 = quantum.custom "RY"(%25) %24 : !quantum.bit
    %26 = arith.extf %22 : f32 to f64
    %out_qubits_36 = quantum.custom "RZ"(%26) %out_qubits_35 : !quantum.bit
    %27 = arith.extf %21 : f32 to f64
    %out_qubits_37 = quantum.custom "RY"(%27) %out_qubits_36 : !quantum.bit
    %28 = arith.extf %20 : f32 to f64
    %out_qubits_38 = quantum.custom "RZ"(%28) %out_qubits_37 : !quantum.bit
    %subview_39 = memref.subview %arg0[0, 5, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 17>>
    %collapse_shape_40 = memref.collapse_shape %subview_39 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 17>> into memref<f32, strided<[], offset: 17>>
    %29 = memref.load %collapse_shape_40[] : memref<f32, strided<[], offset: 17>>
    %subview_41 = memref.subview %arg0[0, 5, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 16>>
    %collapse_shape_42 = memref.collapse_shape %subview_41 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 16>> into memref<f32, strided<[], offset: 16>>
    %30 = memref.load %collapse_shape_42[] : memref<f32, strided<[], offset: 16>>
    %subview_43 = memref.subview %arg0[0, 5, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 15>>
    %collapse_shape_44 = memref.collapse_shape %subview_43 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 15>> into memref<f32, strided<[], offset: 15>>
    %31 = memref.load %collapse_shape_44[] : memref<f32, strided<[], offset: 15>>
    %subview_45 = memref.subview %alloc[5] [1] [1] : memref<8xf32> to memref<1xf32, strided<[1], offset: 5>>
    %collapse_shape_46 = memref.collapse_shape %subview_45 [] : memref<1xf32, strided<[1], offset: 5>> into memref<f32, strided<[], offset: 5>>
    %32 = memref.load %collapse_shape_46[] : memref<f32, strided<[], offset: 5>>
    %33 = quantum.extract %14[ 5] : !quantum.reg -> !quantum.bit
    %34 = arith.extf %32 : f32 to f64
    %out_qubits_47 = quantum.custom "RY"(%34) %33 : !quantum.bit
    %35 = arith.extf %31 : f32 to f64
    %out_qubits_48 = quantum.custom "RZ"(%35) %out_qubits_47 : !quantum.bit
    %36 = arith.extf %30 : f32 to f64
    %out_qubits_49 = quantum.custom "RY"(%36) %out_qubits_48 : !quantum.bit
    %37 = arith.extf %29 : f32 to f64
    %out_qubits_50 = quantum.custom "RZ"(%37) %out_qubits_49 : !quantum.bit
    %subview_51 = memref.subview %arg0[0, 4, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 14>>
    %collapse_shape_52 = memref.collapse_shape %subview_51 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 14>> into memref<f32, strided<[], offset: 14>>
    %38 = memref.load %collapse_shape_52[] : memref<f32, strided<[], offset: 14>>
    %subview_53 = memref.subview %arg0[0, 4, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 13>>
    %collapse_shape_54 = memref.collapse_shape %subview_53 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 13>> into memref<f32, strided<[], offset: 13>>
    %39 = memref.load %collapse_shape_54[] : memref<f32, strided<[], offset: 13>>
    %subview_55 = memref.subview %arg0[0, 4, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 12>>
    %collapse_shape_56 = memref.collapse_shape %subview_55 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 12>> into memref<f32, strided<[], offset: 12>>
    %40 = memref.load %collapse_shape_56[] : memref<f32, strided<[], offset: 12>>
    %subview_57 = memref.subview %alloc[4] [1] [1] : memref<8xf32> to memref<1xf32, strided<[1], offset: 4>>
    %collapse_shape_58 = memref.collapse_shape %subview_57 [] : memref<1xf32, strided<[1], offset: 4>> into memref<f32, strided<[], offset: 4>>
    %41 = memref.load %collapse_shape_58[] : memref<f32, strided<[], offset: 4>>
    %42 = quantum.extract %14[ 4] : !quantum.reg -> !quantum.bit
    %43 = arith.extf %41 : f32 to f64
    %out_qubits_59 = quantum.custom "RY"(%43) %42 : !quantum.bit
    %44 = arith.extf %40 : f32 to f64
    %out_qubits_60 = quantum.custom "RZ"(%44) %out_qubits_59 : !quantum.bit
    %45 = arith.extf %39 : f32 to f64
    %out_qubits_61 = quantum.custom "RY"(%45) %out_qubits_60 : !quantum.bit
    %46 = arith.extf %38 : f32 to f64
    %out_qubits_62 = quantum.custom "RZ"(%46) %out_qubits_61 : !quantum.bit
    %subview_63 = memref.subview %arg0[0, 3, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 11>>
    %collapse_shape_64 = memref.collapse_shape %subview_63 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 11>> into memref<f32, strided<[], offset: 11>>
    %47 = memref.load %collapse_shape_64[] : memref<f32, strided<[], offset: 11>>
    %subview_65 = memref.subview %arg0[0, 3, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 10>>
    %collapse_shape_66 = memref.collapse_shape %subview_65 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 10>> into memref<f32, strided<[], offset: 10>>
    %48 = memref.load %collapse_shape_66[] : memref<f32, strided<[], offset: 10>>
    %subview_67 = memref.subview %arg0[0, 3, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 9>>
    %collapse_shape_68 = memref.collapse_shape %subview_67 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 9>> into memref<f32, strided<[], offset: 9>>
    %49 = memref.load %collapse_shape_68[] : memref<f32, strided<[], offset: 9>>
    %subview_69 = memref.subview %alloc[3] [1] [1] : memref<8xf32> to memref<1xf32, strided<[1], offset: 3>>
    %collapse_shape_70 = memref.collapse_shape %subview_69 [] : memref<1xf32, strided<[1], offset: 3>> into memref<f32, strided<[], offset: 3>>
    %50 = memref.load %collapse_shape_70[] : memref<f32, strided<[], offset: 3>>
    %51 = quantum.extract %14[ 3] : !quantum.reg -> !quantum.bit
    %52 = arith.extf %50 : f32 to f64
    %out_qubits_71 = quantum.custom "RY"(%52) %51 : !quantum.bit
    %53 = arith.extf %49 : f32 to f64
    %out_qubits_72 = quantum.custom "RZ"(%53) %out_qubits_71 : !quantum.bit
    %54 = arith.extf %48 : f32 to f64
    %out_qubits_73 = quantum.custom "RY"(%54) %out_qubits_72 : !quantum.bit
    %55 = arith.extf %47 : f32 to f64
    %out_qubits_74 = quantum.custom "RZ"(%55) %out_qubits_73 : !quantum.bit
    %subview_75 = memref.subview %arg0[0, 2, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 8>>
    %collapse_shape_76 = memref.collapse_shape %subview_75 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 8>> into memref<f32, strided<[], offset: 8>>
    %56 = memref.load %collapse_shape_76[] : memref<f32, strided<[], offset: 8>>
    %subview_77 = memref.subview %arg0[0, 2, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 7>>
    %collapse_shape_78 = memref.collapse_shape %subview_77 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 7>> into memref<f32, strided<[], offset: 7>>
    %57 = memref.load %collapse_shape_78[] : memref<f32, strided<[], offset: 7>>
    %subview_79 = memref.subview %arg0[0, 2, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 6>>
    %collapse_shape_80 = memref.collapse_shape %subview_79 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 6>> into memref<f32, strided<[], offset: 6>>
    %58 = memref.load %collapse_shape_80[] : memref<f32, strided<[], offset: 6>>
    %subview_81 = memref.subview %alloc[2] [1] [1] : memref<8xf32> to memref<1xf32, strided<[1], offset: 2>>
    %collapse_shape_82 = memref.collapse_shape %subview_81 [] : memref<1xf32, strided<[1], offset: 2>> into memref<f32, strided<[], offset: 2>>
    %59 = memref.load %collapse_shape_82[] : memref<f32, strided<[], offset: 2>>
    %60 = quantum.extract %14[ 2] : !quantum.reg -> !quantum.bit
    %61 = arith.extf %59 : f32 to f64
    %out_qubits_83 = quantum.custom "RY"(%61) %60 : !quantum.bit
    %62 = arith.extf %58 : f32 to f64
    %out_qubits_84 = quantum.custom "RZ"(%62) %out_qubits_83 : !quantum.bit
    %63 = arith.extf %57 : f32 to f64
    %out_qubits_85 = quantum.custom "RY"(%63) %out_qubits_84 : !quantum.bit
    %64 = arith.extf %56 : f32 to f64
    %out_qubits_86 = quantum.custom "RZ"(%64) %out_qubits_85 : !quantum.bit
    %subview_87 = memref.subview %arg0[0, 0, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 2>>
    %collapse_shape_88 = memref.collapse_shape %subview_87 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 2>> into memref<f32, strided<[], offset: 2>>
    %65 = memref.load %collapse_shape_88[] : memref<f32, strided<[], offset: 2>>
    %subview_89 = memref.subview %arg0[0, 0, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 1>>
    %collapse_shape_90 = memref.collapse_shape %subview_89 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 1>> into memref<f32, strided<[], offset: 1>>
    %66 = memref.load %collapse_shape_90[] : memref<f32, strided<[], offset: 1>>
    %subview_91 = memref.subview %arg0[0, 0, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1]>>
    %collapse_shape_92 = memref.collapse_shape %subview_91 [] : memref<1x1x1xf32, strided<[24, 3, 1]>> into memref<f32, strided<[]>>
    %67 = memref.load %collapse_shape_92[] : memref<f32, strided<[]>>
    %subview_93 = memref.subview %alloc[0] [1] [1] : memref<8xf32> to memref<1xf32, strided<[1]>>
    %collapse_shape_94 = memref.collapse_shape %subview_93 [] : memref<1xf32, strided<[1]>> into memref<f32>
    %68 = memref.load %collapse_shape_94[] : memref<f32>
    %69 = quantum.extract %14[ 0] : !quantum.reg -> !quantum.bit
    %70 = arith.extf %68 : f32 to f64
    %out_qubits_95 = quantum.custom "RY"(%70) %69 : !quantum.bit
    %71 = arith.extf %67 : f32 to f64
    %out_qubits_96 = quantum.custom "RZ"(%71) %out_qubits_95 : !quantum.bit
    %72 = arith.extf %66 : f32 to f64
    %out_qubits_97 = quantum.custom "RY"(%72) %out_qubits_96 : !quantum.bit
    %73 = arith.extf %65 : f32 to f64
    %out_qubits_98 = quantum.custom "RZ"(%73) %out_qubits_97 : !quantum.bit
    %subview_99 = memref.subview %arg0[0, 1, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 5>>
    %collapse_shape_100 = memref.collapse_shape %subview_99 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 5>> into memref<f32, strided<[], offset: 5>>
    %74 = memref.load %collapse_shape_100[] : memref<f32, strided<[], offset: 5>>
    %subview_101 = memref.subview %arg0[0, 1, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 4>>
    %collapse_shape_102 = memref.collapse_shape %subview_101 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 4>> into memref<f32, strided<[], offset: 4>>
    %75 = memref.load %collapse_shape_102[] : memref<f32, strided<[], offset: 4>>
    %subview_103 = memref.subview %arg0[0, 1, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 3>>
    %collapse_shape_104 = memref.collapse_shape %subview_103 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 3>> into memref<f32, strided<[], offset: 3>>
    %76 = memref.load %collapse_shape_104[] : memref<f32, strided<[], offset: 3>>
    %subview_105 = memref.subview %alloc[1] [1] [1] : memref<8xf32> to memref<1xf32, strided<[1], offset: 1>>
    %collapse_shape_106 = memref.collapse_shape %subview_105 [] : memref<1xf32, strided<[1], offset: 1>> into memref<f32, strided<[], offset: 1>>
    %77 = memref.load %collapse_shape_106[] : memref<f32, strided<[], offset: 1>>
    memref.dealloc %alloc : memref<8xf32>
    %78 = quantum.extract %14[ 1] : !quantum.reg -> !quantum.bit
    %79 = arith.extf %77 : f32 to f64
    %out_qubits_107 = quantum.custom "RY"(%79) %78 : !quantum.bit
    %80 = arith.extf %76 : f32 to f64
    %out_qubits_108 = quantum.custom "RZ"(%80) %out_qubits_107 : !quantum.bit
    %81 = arith.extf %75 : f32 to f64
    %out_qubits_109 = quantum.custom "RY"(%81) %out_qubits_108 : !quantum.bit
    %82 = arith.extf %74 : f32 to f64
    %out_qubits_110 = quantum.custom "RZ"(%82) %out_qubits_109 : !quantum.bit
    %out_qubits_111:2 = quantum.custom "CNOT"() %out_qubits_98, %out_qubits_110 : !quantum.bit, !quantum.bit
    %out_qubits_112:2 = quantum.custom "CNOT"() %out_qubits_111#1, %out_qubits_86 : !quantum.bit, !quantum.bit
    %out_qubits_113:2 = quantum.custom "CNOT"() %out_qubits_112#1, %out_qubits_74 : !quantum.bit, !quantum.bit
    %out_qubits_114:2 = quantum.custom "CNOT"() %out_qubits_113#1, %out_qubits_62 : !quantum.bit, !quantum.bit
    %out_qubits_115:2 = quantum.custom "CNOT"() %out_qubits_114#1, %out_qubits_50 : !quantum.bit, !quantum.bit
    %out_qubits_116:2 = quantum.custom "CNOT"() %out_qubits_115#1, %out_qubits_38 : !quantum.bit, !quantum.bit
    %out_qubits_117:2 = quantum.custom "CNOT"() %out_qubits_116#1, %out_qubits_26 : !quantum.bit, !quantum.bit
    %83 = arith.extf %9 : f32 to f64
    %out_qubits_118 = quantum.custom "RZ"(%83) %out_qubits_117#0 : !quantum.bit
    %84 = arith.extf %8 : f32 to f64
    %out_qubits_119 = quantum.custom "RY"(%84) %out_qubits_118 : !quantum.bit
    %85 = arith.extf %7 : f32 to f64
    %out_qubits_120 = quantum.custom "RZ"(%85) %out_qubits_119 : !quantum.bit
    %subview_121 = memref.subview %arg0[1, 4, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 38>>
    %collapse_shape_122 = memref.collapse_shape %subview_121 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 38>> into memref<f32, strided<[], offset: 38>>
    %86 = memref.load %collapse_shape_122[] : memref<f32, strided<[], offset: 38>>
    %subview_123 = memref.subview %arg0[1, 4, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 37>>
    %collapse_shape_124 = memref.collapse_shape %subview_123 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 37>> into memref<f32, strided<[], offset: 37>>
    %87 = memref.load %collapse_shape_124[] : memref<f32, strided<[], offset: 37>>
    %subview_125 = memref.subview %arg0[1, 4, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 36>>
    %collapse_shape_126 = memref.collapse_shape %subview_125 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 36>> into memref<f32, strided<[], offset: 36>>
    %88 = memref.load %collapse_shape_126[] : memref<f32, strided<[], offset: 36>>
    %89 = arith.extf %88 : f32 to f64
    %out_qubits_127 = quantum.custom "RZ"(%89) %out_qubits_115#0 : !quantum.bit
    %90 = arith.extf %87 : f32 to f64
    %out_qubits_128 = quantum.custom "RY"(%90) %out_qubits_127 : !quantum.bit
    %91 = arith.extf %86 : f32 to f64
    %out_qubits_129 = quantum.custom "RZ"(%91) %out_qubits_128 : !quantum.bit
    %subview_130 = memref.subview %arg0[1, 0, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 26>>
    %collapse_shape_131 = memref.collapse_shape %subview_130 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 26>> into memref<f32, strided<[], offset: 26>>
    %92 = memref.load %collapse_shape_131[] : memref<f32, strided<[], offset: 26>>
    %subview_132 = memref.subview %arg0[1, 0, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 25>>
    %collapse_shape_133 = memref.collapse_shape %subview_132 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 25>> into memref<f32, strided<[], offset: 25>>
    %93 = memref.load %collapse_shape_133[] : memref<f32, strided<[], offset: 25>>
    %subview_134 = memref.subview %arg0[1, 0, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 24>>
    %collapse_shape_135 = memref.collapse_shape %subview_134 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 24>> into memref<f32, strided<[], offset: 24>>
    %94 = memref.load %collapse_shape_135[] : memref<f32, strided<[], offset: 24>>
    %out_qubits_136:2 = quantum.custom "CNOT"() %out_qubits_117#1, %out_qubits_111#0 : !quantum.bit, !quantum.bit
    %95 = arith.extf %94 : f32 to f64
    %out_qubits_137 = quantum.custom "RZ"(%95) %out_qubits_136#1 : !quantum.bit
    %96 = arith.extf %93 : f32 to f64
    %out_qubits_138 = quantum.custom "RY"(%96) %out_qubits_137 : !quantum.bit
    %97 = arith.extf %92 : f32 to f64
    %out_qubits_139 = quantum.custom "RZ"(%97) %out_qubits_138 : !quantum.bit
    %subview_140 = memref.subview %arg0[1, 2, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 32>>
    %collapse_shape_141 = memref.collapse_shape %subview_140 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 32>> into memref<f32, strided<[], offset: 32>>
    %98 = memref.load %collapse_shape_141[] : memref<f32, strided<[], offset: 32>>
    %subview_142 = memref.subview %arg0[1, 2, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 31>>
    %collapse_shape_143 = memref.collapse_shape %subview_142 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 31>> into memref<f32, strided<[], offset: 31>>
    %99 = memref.load %collapse_shape_143[] : memref<f32, strided<[], offset: 31>>
    %subview_144 = memref.subview %arg0[1, 2, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 30>>
    %collapse_shape_145 = memref.collapse_shape %subview_144 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 30>> into memref<f32, strided<[], offset: 30>>
    %100 = memref.load %collapse_shape_145[] : memref<f32, strided<[], offset: 30>>
    %101 = arith.extf %100 : f32 to f64
    %out_qubits_146 = quantum.custom "RZ"(%101) %out_qubits_113#0 : !quantum.bit
    %102 = arith.extf %99 : f32 to f64
    %out_qubits_147 = quantum.custom "RY"(%102) %out_qubits_146 : !quantum.bit
    %103 = arith.extf %98 : f32 to f64
    %out_qubits_148 = quantum.custom "RZ"(%103) %out_qubits_147 : !quantum.bit
    %out_qubits_149:2 = quantum.custom "CNOT"() %out_qubits_139, %out_qubits_148 : !quantum.bit, !quantum.bit
    %out_qubits_150:2 = quantum.custom "CNOT"() %out_qubits_149#1, %out_qubits_129 : !quantum.bit, !quantum.bit
    %out_qubits_151:2 = quantum.custom "CNOT"() %out_qubits_150#1, %out_qubits_120 : !quantum.bit, !quantum.bit
    %out_qubits_152:2 = quantum.custom "CNOT"() %out_qubits_151#1, %out_qubits_149#0 : !quantum.bit, !quantum.bit
    %104 = arith.extf %6 : f32 to f64
    %out_qubits_153 = quantum.custom "RZ"(%104) %out_qubits_152#1 : !quantum.bit
    %105 = arith.extf %5 : f32 to f64
    %out_qubits_154 = quantum.custom "RY"(%105) %out_qubits_153 : !quantum.bit
    %106 = arith.extf %4 : f32 to f64
    %out_qubits_155 = quantum.custom "RZ"(%106) %out_qubits_154 : !quantum.bit
    %subview_156 = memref.subview %arg0[2, 3, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 59>>
    %collapse_shape_157 = memref.collapse_shape %subview_156 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 59>> into memref<f32, strided<[], offset: 59>>
    %107 = memref.load %collapse_shape_157[] : memref<f32, strided<[], offset: 59>>
    %subview_158 = memref.subview %arg0[2, 3, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 58>>
    %collapse_shape_159 = memref.collapse_shape %subview_158 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 58>> into memref<f32, strided<[], offset: 58>>
    %108 = memref.load %collapse_shape_159[] : memref<f32, strided<[], offset: 58>>
    %subview_160 = memref.subview %arg0[2, 3, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 57>>
    %collapse_shape_161 = memref.collapse_shape %subview_160 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 57>> into memref<f32, strided<[], offset: 57>>
    %109 = memref.load %collapse_shape_161[] : memref<f32, strided<[], offset: 57>>
    %subview_162 = memref.subview %arg0[1, 5, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 41>>
    %collapse_shape_163 = memref.collapse_shape %subview_162 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 41>> into memref<f32, strided<[], offset: 41>>
    %110 = memref.load %collapse_shape_163[] : memref<f32, strided<[], offset: 41>>
    %subview_164 = memref.subview %arg0[1, 5, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 40>>
    %collapse_shape_165 = memref.collapse_shape %subview_164 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 40>> into memref<f32, strided<[], offset: 40>>
    %111 = memref.load %collapse_shape_165[] : memref<f32, strided<[], offset: 40>>
    %subview_166 = memref.subview %arg0[1, 5, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 39>>
    %collapse_shape_167 = memref.collapse_shape %subview_166 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 39>> into memref<f32, strided<[], offset: 39>>
    %112 = memref.load %collapse_shape_167[] : memref<f32, strided<[], offset: 39>>
    %113 = arith.extf %112 : f32 to f64
    %out_qubits_168 = quantum.custom "RZ"(%113) %out_qubits_116#0 : !quantum.bit
    %114 = arith.extf %111 : f32 to f64
    %out_qubits_169 = quantum.custom "RY"(%114) %out_qubits_168 : !quantum.bit
    %115 = arith.extf %110 : f32 to f64
    %out_qubits_170 = quantum.custom "RZ"(%115) %out_qubits_169 : !quantum.bit
    %subview_171 = memref.subview %arg0[1, 1, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 29>>
    %collapse_shape_172 = memref.collapse_shape %subview_171 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 29>> into memref<f32, strided<[], offset: 29>>
    %116 = memref.load %collapse_shape_172[] : memref<f32, strided<[], offset: 29>>
    %subview_173 = memref.subview %arg0[1, 1, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 28>>
    %collapse_shape_174 = memref.collapse_shape %subview_173 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 28>> into memref<f32, strided<[], offset: 28>>
    %117 = memref.load %collapse_shape_174[] : memref<f32, strided<[], offset: 28>>
    %subview_175 = memref.subview %arg0[1, 1, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 27>>
    %collapse_shape_176 = memref.collapse_shape %subview_175 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 27>> into memref<f32, strided<[], offset: 27>>
    %118 = memref.load %collapse_shape_176[] : memref<f32, strided<[], offset: 27>>
    %119 = arith.extf %118 : f32 to f64
    %out_qubits_177 = quantum.custom "RZ"(%119) %out_qubits_112#0 : !quantum.bit
    %120 = arith.extf %117 : f32 to f64
    %out_qubits_178 = quantum.custom "RY"(%120) %out_qubits_177 : !quantum.bit
    %121 = arith.extf %116 : f32 to f64
    %out_qubits_179 = quantum.custom "RZ"(%121) %out_qubits_178 : !quantum.bit
    %subview_180 = memref.subview %arg0[1, 3, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 35>>
    %collapse_shape_181 = memref.collapse_shape %subview_180 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 35>> into memref<f32, strided<[], offset: 35>>
    %122 = memref.load %collapse_shape_181[] : memref<f32, strided<[], offset: 35>>
    %subview_182 = memref.subview %arg0[1, 3, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 34>>
    %collapse_shape_183 = memref.collapse_shape %subview_182 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 34>> into memref<f32, strided<[], offset: 34>>
    %123 = memref.load %collapse_shape_183[] : memref<f32, strided<[], offset: 34>>
    %subview_184 = memref.subview %arg0[1, 3, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 33>>
    %collapse_shape_185 = memref.collapse_shape %subview_184 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 33>> into memref<f32, strided<[], offset: 33>>
    %124 = memref.load %collapse_shape_185[] : memref<f32, strided<[], offset: 33>>
    %125 = arith.extf %124 : f32 to f64
    %out_qubits_186 = quantum.custom "RZ"(%125) %out_qubits_114#0 : !quantum.bit
    %126 = arith.extf %123 : f32 to f64
    %out_qubits_187 = quantum.custom "RY"(%126) %out_qubits_186 : !quantum.bit
    %127 = arith.extf %122 : f32 to f64
    %out_qubits_188 = quantum.custom "RZ"(%127) %out_qubits_187 : !quantum.bit
    %out_qubits_189:2 = quantum.custom "CNOT"() %out_qubits_179, %out_qubits_188 : !quantum.bit, !quantum.bit
    %out_qubits_190:2 = quantum.custom "CNOT"() %out_qubits_189#1, %out_qubits_170 : !quantum.bit, !quantum.bit
    %128 = arith.extf %109 : f32 to f64
    %out_qubits_191 = quantum.custom "RZ"(%128) %out_qubits_190#0 : !quantum.bit
    %129 = arith.extf %108 : f32 to f64
    %out_qubits_192 = quantum.custom "RY"(%129) %out_qubits_191 : !quantum.bit
    %130 = arith.extf %107 : f32 to f64
    %out_qubits_193 = quantum.custom "RZ"(%130) %out_qubits_192 : !quantum.bit
    %out_qubits_194:2 = quantum.custom "CNOT"() %out_qubits_155, %out_qubits_193 : !quantum.bit, !quantum.bit
    %subview_195 = memref.subview %arg0[2, 2, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 56>>
    %collapse_shape_196 = memref.collapse_shape %subview_195 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 56>> into memref<f32, strided<[], offset: 56>>
    %131 = memref.load %collapse_shape_196[] : memref<f32, strided<[], offset: 56>>
    %subview_197 = memref.subview %arg0[2, 2, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 55>>
    %collapse_shape_198 = memref.collapse_shape %subview_197 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 55>> into memref<f32, strided<[], offset: 55>>
    %132 = memref.load %collapse_shape_198[] : memref<f32, strided<[], offset: 55>>
    %subview_199 = memref.subview %arg0[2, 2, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 54>>
    %collapse_shape_200 = memref.collapse_shape %subview_199 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 54>> into memref<f32, strided<[], offset: 54>>
    %133 = memref.load %collapse_shape_200[] : memref<f32, strided<[], offset: 54>>
    %134 = arith.extf %133 : f32 to f64
    %out_qubits_201 = quantum.custom "RZ"(%134) %out_qubits_150#0 : !quantum.bit
    %135 = arith.extf %132 : f32 to f64
    %out_qubits_202 = quantum.custom "RY"(%135) %out_qubits_201 : !quantum.bit
    %136 = arith.extf %131 : f32 to f64
    %out_qubits_203 = quantum.custom "RZ"(%136) %out_qubits_202 : !quantum.bit
    %subview_204 = memref.subview %arg0[2, 5, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 65>>
    %collapse_shape_205 = memref.collapse_shape %subview_204 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 65>> into memref<f32, strided<[], offset: 65>>
    %137 = memref.load %collapse_shape_205[] : memref<f32, strided<[], offset: 65>>
    %subview_206 = memref.subview %arg0[2, 5, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 64>>
    %collapse_shape_207 = memref.collapse_shape %subview_206 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 64>> into memref<f32, strided<[], offset: 64>>
    %138 = memref.load %collapse_shape_207[] : memref<f32, strided<[], offset: 64>>
    %subview_208 = memref.subview %arg0[2, 5, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 63>>
    %collapse_shape_209 = memref.collapse_shape %subview_208 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 63>> into memref<f32, strided<[], offset: 63>>
    %139 = memref.load %collapse_shape_209[] : memref<f32, strided<[], offset: 63>>
    %subview_210 = memref.subview %arg0[1, 7, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 47>>
    %collapse_shape_211 = memref.collapse_shape %subview_210 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 47>> into memref<f32, strided<[], offset: 47>>
    %140 = memref.load %collapse_shape_211[] : memref<f32, strided<[], offset: 47>>
    %subview_212 = memref.subview %arg0[1, 7, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 46>>
    %collapse_shape_213 = memref.collapse_shape %subview_212 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 46>> into memref<f32, strided<[], offset: 46>>
    %141 = memref.load %collapse_shape_213[] : memref<f32, strided<[], offset: 46>>
    %subview_214 = memref.subview %arg0[1, 7, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 45>>
    %collapse_shape_215 = memref.collapse_shape %subview_214 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 45>> into memref<f32, strided<[], offset: 45>>
    %142 = memref.load %collapse_shape_215[] : memref<f32, strided<[], offset: 45>>
    %143 = arith.extf %142 : f32 to f64
    %out_qubits_216 = quantum.custom "RZ"(%143) %out_qubits_136#0 : !quantum.bit
    %144 = arith.extf %141 : f32 to f64
    %out_qubits_217 = quantum.custom "RY"(%144) %out_qubits_216 : !quantum.bit
    %145 = arith.extf %140 : f32 to f64
    %out_qubits_218 = quantum.custom "RZ"(%145) %out_qubits_217 : !quantum.bit
    %out_qubits_219:2 = quantum.custom "CNOT"() %out_qubits_190#1, %out_qubits_218 : !quantum.bit, !quantum.bit
    %146 = arith.extf %139 : f32 to f64
    %out_qubits_220 = quantum.custom "RZ"(%146) %out_qubits_219#0 : !quantum.bit
    %147 = arith.extf %138 : f32 to f64
    %out_qubits_221 = quantum.custom "RY"(%147) %out_qubits_220 : !quantum.bit
    %148 = arith.extf %137 : f32 to f64
    %out_qubits_222 = quantum.custom "RZ"(%148) %out_qubits_221 : !quantum.bit
    %out_qubits_223:2 = quantum.custom "CNOT"() %out_qubits_203, %out_qubits_222 : !quantum.bit, !quantum.bit
    %out_qubits_224:2 = quantum.custom "CNOT"() %out_qubits_223#1, %out_qubits_194#0 : !quantum.bit, !quantum.bit
    %149 = arith.extf %3 : f32 to f64
    %out_qubits_225 = quantum.custom "RZ"(%149) %out_qubits_224#1 : !quantum.bit
    %150 = arith.extf %2 : f32 to f64
    %out_qubits_226 = quantum.custom "RY"(%150) %out_qubits_225 : !quantum.bit
    %151 = arith.extf %1 : f32 to f64
    %out_qubits_227 = quantum.custom "RZ"(%151) %out_qubits_226 : !quantum.bit
    %subview_228 = memref.subview %arg0[3, 4, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 86>>
    %collapse_shape_229 = memref.collapse_shape %subview_228 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 86>> into memref<f32, strided<[], offset: 86>>
    %152 = memref.load %collapse_shape_229[] : memref<f32, strided<[], offset: 86>>
    %subview_230 = memref.subview %arg0[3, 4, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 85>>
    %collapse_shape_231 = memref.collapse_shape %subview_230 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 85>> into memref<f32, strided<[], offset: 85>>
    %153 = memref.load %collapse_shape_231[] : memref<f32, strided<[], offset: 85>>
    %subview_232 = memref.subview %arg0[3, 4, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 84>>
    %collapse_shape_233 = memref.collapse_shape %subview_232 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 84>> into memref<f32, strided<[], offset: 84>>
    %154 = memref.load %collapse_shape_233[] : memref<f32, strided<[], offset: 84>>
    %subview_234 = memref.subview %arg0[2, 7, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 71>>
    %collapse_shape_235 = memref.collapse_shape %subview_234 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 71>> into memref<f32, strided<[], offset: 71>>
    %155 = memref.load %collapse_shape_235[] : memref<f32, strided<[], offset: 71>>
    %subview_236 = memref.subview %arg0[2, 7, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 70>>
    %collapse_shape_237 = memref.collapse_shape %subview_236 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 70>> into memref<f32, strided<[], offset: 70>>
    %156 = memref.load %collapse_shape_237[] : memref<f32, strided<[], offset: 70>>
    %subview_238 = memref.subview %arg0[2, 7, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 69>>
    %collapse_shape_239 = memref.collapse_shape %subview_238 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 69>> into memref<f32, strided<[], offset: 69>>
    %157 = memref.load %collapse_shape_239[] : memref<f32, strided<[], offset: 69>>
    %out_qubits_240:2 = quantum.custom "CNOT"() %out_qubits_219#1, %out_qubits_189#0 : !quantum.bit, !quantum.bit
    %158 = arith.extf %157 : f32 to f64
    %out_qubits_241 = quantum.custom "RZ"(%158) %out_qubits_240#0 : !quantum.bit
    %159 = arith.extf %156 : f32 to f64
    %out_qubits_242 = quantum.custom "RY"(%159) %out_qubits_241 : !quantum.bit
    %160 = arith.extf %155 : f32 to f64
    %out_qubits_243 = quantum.custom "RZ"(%160) %out_qubits_242 : !quantum.bit
    %subview_244 = memref.subview %arg0[2, 1, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 53>>
    %collapse_shape_245 = memref.collapse_shape %subview_244 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 53>> into memref<f32, strided<[], offset: 53>>
    %161 = memref.load %collapse_shape_245[] : memref<f32, strided<[], offset: 53>>
    %subview_246 = memref.subview %arg0[2, 1, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 52>>
    %collapse_shape_247 = memref.collapse_shape %subview_246 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 52>> into memref<f32, strided<[], offset: 52>>
    %162 = memref.load %collapse_shape_247[] : memref<f32, strided<[], offset: 52>>
    %subview_248 = memref.subview %arg0[2, 1, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 51>>
    %collapse_shape_249 = memref.collapse_shape %subview_248 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 51>> into memref<f32, strided<[], offset: 51>>
    %163 = memref.load %collapse_shape_249[] : memref<f32, strided<[], offset: 51>>
    %164 = arith.extf %163 : f32 to f64
    %out_qubits_250 = quantum.custom "RZ"(%164) %out_qubits_240#1 : !quantum.bit
    %165 = arith.extf %162 : f32 to f64
    %out_qubits_251 = quantum.custom "RY"(%165) %out_qubits_250 : !quantum.bit
    %166 = arith.extf %161 : f32 to f64
    %out_qubits_252 = quantum.custom "RZ"(%166) %out_qubits_251 : !quantum.bit
    %subview_253 = memref.subview %arg0[2, 4, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 62>>
    %collapse_shape_254 = memref.collapse_shape %subview_253 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 62>> into memref<f32, strided<[], offset: 62>>
    %167 = memref.load %collapse_shape_254[] : memref<f32, strided<[], offset: 62>>
    %subview_255 = memref.subview %arg0[2, 4, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 61>>
    %collapse_shape_256 = memref.collapse_shape %subview_255 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 61>> into memref<f32, strided<[], offset: 61>>
    %168 = memref.load %collapse_shape_256[] : memref<f32, strided<[], offset: 61>>
    %subview_257 = memref.subview %arg0[2, 4, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 60>>
    %collapse_shape_258 = memref.collapse_shape %subview_257 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 60>> into memref<f32, strided<[], offset: 60>>
    %169 = memref.load %collapse_shape_258[] : memref<f32, strided<[], offset: 60>>
    %170 = arith.extf %169 : f32 to f64
    %out_qubits_259 = quantum.custom "RZ"(%170) %out_qubits_151#0 : !quantum.bit
    %171 = arith.extf %168 : f32 to f64
    %out_qubits_260 = quantum.custom "RY"(%171) %out_qubits_259 : !quantum.bit
    %172 = arith.extf %167 : f32 to f64
    %out_qubits_261 = quantum.custom "RZ"(%172) %out_qubits_260 : !quantum.bit
    %out_qubits_262:2 = quantum.custom "CNOT"() %out_qubits_252, %out_qubits_261 : !quantum.bit, !quantum.bit
    %out_qubits_263:2 = quantum.custom "CNOT"() %out_qubits_262#1, %out_qubits_243 : !quantum.bit, !quantum.bit
    %173 = arith.extf %154 : f32 to f64
    %out_qubits_264 = quantum.custom "RZ"(%173) %out_qubits_263#0 : !quantum.bit
    %174 = arith.extf %153 : f32 to f64
    %out_qubits_265 = quantum.custom "RY"(%174) %out_qubits_264 : !quantum.bit
    %175 = arith.extf %152 : f32 to f64
    %out_qubits_266 = quantum.custom "RZ"(%175) %out_qubits_265 : !quantum.bit
    %out_qubits_267:2 = quantum.custom "CNOT"() %out_qubits_227, %out_qubits_266 : !quantum.bit, !quantum.bit
    %out_qubits_268:2 = quantum.custom "CNOT"() %out_qubits_267#1, %out_qubits_267#0 : !quantum.bit, !quantum.bit
    %subview_269 = memref.subview %arg0[3, 1, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 77>>
    %collapse_shape_270 = memref.collapse_shape %subview_269 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 77>> into memref<f32, strided<[], offset: 77>>
    %176 = memref.load %collapse_shape_270[] : memref<f32, strided<[], offset: 77>>
    %subview_271 = memref.subview %arg0[3, 1, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 76>>
    %collapse_shape_272 = memref.collapse_shape %subview_271 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 76>> into memref<f32, strided<[], offset: 76>>
    %177 = memref.load %collapse_shape_272[] : memref<f32, strided<[], offset: 76>>
    %subview_273 = memref.subview %arg0[3, 1, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 75>>
    %collapse_shape_274 = memref.collapse_shape %subview_273 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 75>> into memref<f32, strided<[], offset: 75>>
    %178 = memref.load %collapse_shape_274[] : memref<f32, strided<[], offset: 75>>
    %subview_275 = memref.subview %arg0[2, 6, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 68>>
    %collapse_shape_276 = memref.collapse_shape %subview_275 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 68>> into memref<f32, strided<[], offset: 68>>
    %179 = memref.load %collapse_shape_276[] : memref<f32, strided<[], offset: 68>>
    %subview_277 = memref.subview %arg0[2, 6, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 67>>
    %collapse_shape_278 = memref.collapse_shape %subview_277 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 67>> into memref<f32, strided<[], offset: 67>>
    %180 = memref.load %collapse_shape_278[] : memref<f32, strided<[], offset: 67>>
    %subview_279 = memref.subview %arg0[2, 6, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 66>>
    %collapse_shape_280 = memref.collapse_shape %subview_279 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 66>> into memref<f32, strided<[], offset: 66>>
    %181 = memref.load %collapse_shape_280[] : memref<f32, strided<[], offset: 66>>
    %182 = arith.extf %181 : f32 to f64
    %out_qubits_281 = quantum.custom "RZ"(%182) %out_qubits_152#0 : !quantum.bit
    %183 = arith.extf %180 : f32 to f64
    %out_qubits_282 = quantum.custom "RY"(%183) %out_qubits_281 : !quantum.bit
    %184 = arith.extf %179 : f32 to f64
    %out_qubits_283 = quantum.custom "RZ"(%184) %out_qubits_282 : !quantum.bit
    %out_qubits_284:2 = quantum.custom "CNOT"() %out_qubits_194#1, %out_qubits_283 : !quantum.bit, !quantum.bit
    %out_qubits_285:2 = quantum.custom "CNOT"() %out_qubits_284#1, %out_qubits_262#0 : !quantum.bit, !quantum.bit
    %185 = arith.extf %178 : f32 to f64
    %out_qubits_286 = quantum.custom "RZ"(%185) %out_qubits_285#1 : !quantum.bit
    %186 = arith.extf %177 : f32 to f64
    %out_qubits_287 = quantum.custom "RY"(%186) %out_qubits_286 : !quantum.bit
    %187 = arith.extf %176 : f32 to f64
    %out_qubits_288 = quantum.custom "RZ"(%187) %out_qubits_287 : !quantum.bit
    %subview_289 = memref.subview %arg0[3, 5, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 89>>
    %collapse_shape_290 = memref.collapse_shape %subview_289 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 89>> into memref<f32, strided<[], offset: 89>>
    %188 = memref.load %collapse_shape_290[] : memref<f32, strided<[], offset: 89>>
    %subview_291 = memref.subview %arg0[3, 5, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 88>>
    %collapse_shape_292 = memref.collapse_shape %subview_291 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 88>> into memref<f32, strided<[], offset: 88>>
    %189 = memref.load %collapse_shape_292[] : memref<f32, strided<[], offset: 88>>
    %subview_293 = memref.subview %arg0[3, 5, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 87>>
    %collapse_shape_294 = memref.collapse_shape %subview_293 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 87>> into memref<f32, strided<[], offset: 87>>
    %190 = memref.load %collapse_shape_294[] : memref<f32, strided<[], offset: 87>>
    %191 = arith.extf %190 : f32 to f64
    %out_qubits_295 = quantum.custom "RZ"(%191) %out_qubits_224#0 : !quantum.bit
    %192 = arith.extf %189 : f32 to f64
    %out_qubits_296 = quantum.custom "RY"(%192) %out_qubits_295 : !quantum.bit
    %193 = arith.extf %188 : f32 to f64
    %out_qubits_297 = quantum.custom "RZ"(%193) %out_qubits_296 : !quantum.bit
    %out_qubits_298:2 = quantum.custom "CNOT"() %out_qubits_288, %out_qubits_297 : !quantum.bit, !quantum.bit
    %out_qubits_299:2 = quantum.custom "CNOT"() %out_qubits_298#1, %out_qubits_298#0 : !quantum.bit, !quantum.bit
    %subview_300 = memref.subview %arg0[3, 2, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 80>>
    %collapse_shape_301 = memref.collapse_shape %subview_300 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 80>> into memref<f32, strided<[], offset: 80>>
    %194 = memref.load %collapse_shape_301[] : memref<f32, strided<[], offset: 80>>
    %subview_302 = memref.subview %arg0[3, 2, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 79>>
    %collapse_shape_303 = memref.collapse_shape %subview_302 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 79>> into memref<f32, strided<[], offset: 79>>
    %195 = memref.load %collapse_shape_303[] : memref<f32, strided<[], offset: 79>>
    %subview_304 = memref.subview %arg0[3, 2, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 78>>
    %collapse_shape_305 = memref.collapse_shape %subview_304 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 78>> into memref<f32, strided<[], offset: 78>>
    %196 = memref.load %collapse_shape_305[] : memref<f32, strided<[], offset: 78>>
    %out_qubits_306:2 = quantum.custom "CNOT"() %out_qubits_263#1, %out_qubits_223#0 : !quantum.bit, !quantum.bit
    %197 = arith.extf %196 : f32 to f64
    %out_qubits_307 = quantum.custom "RZ"(%197) %out_qubits_306#1 : !quantum.bit
    %198 = arith.extf %195 : f32 to f64
    %out_qubits_308 = quantum.custom "RY"(%198) %out_qubits_307 : !quantum.bit
    %199 = arith.extf %194 : f32 to f64
    %out_qubits_309 = quantum.custom "RZ"(%199) %out_qubits_308 : !quantum.bit
    %subview_310 = memref.subview %arg0[3, 6, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 92>>
    %collapse_shape_311 = memref.collapse_shape %subview_310 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 92>> into memref<f32, strided<[], offset: 92>>
    %200 = memref.load %collapse_shape_311[] : memref<f32, strided<[], offset: 92>>
    %subview_312 = memref.subview %arg0[3, 6, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 91>>
    %collapse_shape_313 = memref.collapse_shape %subview_312 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 91>> into memref<f32, strided<[], offset: 91>>
    %201 = memref.load %collapse_shape_313[] : memref<f32, strided<[], offset: 91>>
    %subview_314 = memref.subview %arg0[3, 6, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 90>>
    %collapse_shape_315 = memref.collapse_shape %subview_314 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 90>> into memref<f32, strided<[], offset: 90>>
    %202 = memref.load %collapse_shape_315[] : memref<f32, strided<[], offset: 90>>
    %203 = arith.extf %202 : f32 to f64
    %out_qubits_316 = quantum.custom "RZ"(%203) %out_qubits_285#0 : !quantum.bit
    %204 = arith.extf %201 : f32 to f64
    %out_qubits_317 = quantum.custom "RY"(%204) %out_qubits_316 : !quantum.bit
    %205 = arith.extf %200 : f32 to f64
    %out_qubits_318 = quantum.custom "RZ"(%205) %out_qubits_317 : !quantum.bit
    %out_qubits_319:2 = quantum.custom "CNOT"() %out_qubits_309, %out_qubits_318 : !quantum.bit, !quantum.bit
    %out_qubits_320:2 = quantum.custom "CNOT"() %out_qubits_319#1, %out_qubits_319#0 : !quantum.bit, !quantum.bit
    %subview_321 = memref.subview %arg0[3, 3, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 83>>
    %collapse_shape_322 = memref.collapse_shape %subview_321 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 83>> into memref<f32, strided<[], offset: 83>>
    %206 = memref.load %collapse_shape_322[] : memref<f32, strided<[], offset: 83>>
    %subview_323 = memref.subview %arg0[3, 3, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 82>>
    %collapse_shape_324 = memref.collapse_shape %subview_323 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 82>> into memref<f32, strided<[], offset: 82>>
    %207 = memref.load %collapse_shape_324[] : memref<f32, strided<[], offset: 82>>
    %subview_325 = memref.subview %arg0[3, 3, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 81>>
    %collapse_shape_326 = memref.collapse_shape %subview_325 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 81>> into memref<f32, strided<[], offset: 81>>
    %208 = memref.load %collapse_shape_326[] : memref<f32, strided<[], offset: 81>>
    %209 = arith.extf %208 : f32 to f64
    %out_qubits_327 = quantum.custom "RZ"(%209) %out_qubits_284#0 : !quantum.bit
    %210 = arith.extf %207 : f32 to f64
    %out_qubits_328 = quantum.custom "RY"(%210) %out_qubits_327 : !quantum.bit
    %211 = arith.extf %206 : f32 to f64
    %out_qubits_329 = quantum.custom "RZ"(%211) %out_qubits_328 : !quantum.bit
    %subview_330 = memref.subview %arg0[3, 7, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 95>>
    %collapse_shape_331 = memref.collapse_shape %subview_330 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 95>> into memref<f32, strided<[], offset: 95>>
    %212 = memref.load %collapse_shape_331[] : memref<f32, strided<[], offset: 95>>
    %subview_332 = memref.subview %arg0[3, 7, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 94>>
    %collapse_shape_333 = memref.collapse_shape %subview_332 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 94>> into memref<f32, strided<[], offset: 94>>
    %213 = memref.load %collapse_shape_333[] : memref<f32, strided<[], offset: 94>>
    %subview_334 = memref.subview %arg0[3, 7, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 93>>
    %collapse_shape_335 = memref.collapse_shape %subview_334 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 93>> into memref<f32, strided<[], offset: 93>>
    %214 = memref.load %collapse_shape_335[] : memref<f32, strided<[], offset: 93>>
    %215 = arith.extf %214 : f32 to f64
    %out_qubits_336 = quantum.custom "RZ"(%215) %out_qubits_306#0 : !quantum.bit
    %216 = arith.extf %213 : f32 to f64
    %out_qubits_337 = quantum.custom "RY"(%216) %out_qubits_336 : !quantum.bit
    %217 = arith.extf %212 : f32 to f64
    %out_qubits_338 = quantum.custom "RZ"(%217) %out_qubits_337 : !quantum.bit
    %out_qubits_339:2 = quantum.custom "CNOT"() %out_qubits_329, %out_qubits_338 : !quantum.bit, !quantum.bit
    %out_qubits_340:2 = quantum.custom "CNOT"() %out_qubits_339#1, %out_qubits_339#0 : !quantum.bit, !quantum.bit
    %218 = quantum.namedobs %out_qubits_268#1[ PauliZ] : !quantum.obs
    %219 = quantum.expval %218 : f64
    %alloc_341 = memref.alloc() {alignment = 64 : i64} : memref<f64>
    memref.store %219, %alloc_341[] : memref<f64>
    %220 = quantum.insert %14[ 0], %out_qubits_268#1 : !quantum.reg, !quantum.bit
    %221 = quantum.insert %220[ 1], %out_qubits_299#1 : !quantum.reg, !quantum.bit
    %222 = quantum.insert %221[ 2], %out_qubits_320#1 : !quantum.reg, !quantum.bit
    %223 = quantum.insert %222[ 3], %out_qubits_340#1 : !quantum.reg, !quantum.bit
    %224 = quantum.insert %223[ 4], %out_qubits_268#0 : !quantum.reg, !quantum.bit
    %225 = quantum.insert %224[ 5], %out_qubits_299#0 : !quantum.reg, !quantum.bit
    %226 = quantum.insert %225[ 6], %out_qubits_320#0 : !quantum.reg, !quantum.bit
    %227 = quantum.insert %226[ 7], %out_qubits_340#0 : !quantum.reg, !quantum.bit
    quantum.dealloc %227 : !quantum.reg
    quantum.device_release
    return %alloc_341 : memref<f64>
  }
  func.func private @qnode_forward_0.adjoint(%arg0: memref<4x8x3xf32>, %arg1: memref<8xf32>, %arg2: index) -> memref<?xf64> {
    %alloc = memref.alloc(%arg2) : memref<?xf64>
    gradient.adjoint @qnode_forward_0.nodealloc(%arg0, %arg1) size(%arg2) in(%alloc : memref<?xf64>) : (memref<4x8x3xf32>, memref<8xf32>) -> ()
    return %alloc : memref<?xf64>
  }
  func.func private @qnode_forward_0.nodealloc(%arg0: memref<4x8x3xf32>, %arg1: memref<8xf32>) -> (!quantum.reg, f64) {
    %c0_i64 = arith.constant 0 : i64
    %0 = memref.get_global @__constant_xf32 : memref<f32>
    %subview = memref.subview %arg0[3, 0, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 74>>
    %collapse_shape = memref.collapse_shape %subview [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 74>> into memref<f32, strided<[], offset: 74>>
    %1 = memref.load %collapse_shape[] : memref<f32, strided<[], offset: 74>>
    %subview_0 = memref.subview %arg0[3, 0, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 73>>
    %collapse_shape_1 = memref.collapse_shape %subview_0 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 73>> into memref<f32, strided<[], offset: 73>>
    %2 = memref.load %collapse_shape_1[] : memref<f32, strided<[], offset: 73>>
    %subview_2 = memref.subview %arg0[3, 0, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 72>>
    %collapse_shape_3 = memref.collapse_shape %subview_2 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 72>> into memref<f32, strided<[], offset: 72>>
    %3 = memref.load %collapse_shape_3[] : memref<f32, strided<[], offset: 72>>
    %subview_4 = memref.subview %arg0[2, 0, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 50>>
    %collapse_shape_5 = memref.collapse_shape %subview_4 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 50>> into memref<f32, strided<[], offset: 50>>
    %4 = memref.load %collapse_shape_5[] : memref<f32, strided<[], offset: 50>>
    %subview_6 = memref.subview %arg0[2, 0, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 49>>
    %collapse_shape_7 = memref.collapse_shape %subview_6 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 49>> into memref<f32, strided<[], offset: 49>>
    %5 = memref.load %collapse_shape_7[] : memref<f32, strided<[], offset: 49>>
    %subview_8 = memref.subview %arg0[2, 0, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 48>>
    %collapse_shape_9 = memref.collapse_shape %subview_8 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 48>> into memref<f32, strided<[], offset: 48>>
    %6 = memref.load %collapse_shape_9[] : memref<f32, strided<[], offset: 48>>
    %subview_10 = memref.subview %arg0[1, 6, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 44>>
    %collapse_shape_11 = memref.collapse_shape %subview_10 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 44>> into memref<f32, strided<[], offset: 44>>
    %7 = memref.load %collapse_shape_11[] : memref<f32, strided<[], offset: 44>>
    %subview_12 = memref.subview %arg0[1, 6, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 43>>
    %collapse_shape_13 = memref.collapse_shape %subview_12 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 43>> into memref<f32, strided<[], offset: 43>>
    %8 = memref.load %collapse_shape_13[] : memref<f32, strided<[], offset: 43>>
    %subview_14 = memref.subview %arg0[1, 6, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 42>>
    %collapse_shape_15 = memref.collapse_shape %subview_14 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 42>> into memref<f32, strided<[], offset: 42>>
    %9 = memref.load %collapse_shape_15[] : memref<f32, strided<[], offset: 42>>
    %subview_16 = memref.subview %arg0[0, 7, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 23>>
    %collapse_shape_17 = memref.collapse_shape %subview_16 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 23>> into memref<f32, strided<[], offset: 23>>
    %10 = memref.load %collapse_shape_17[] : memref<f32, strided<[], offset: 23>>
    %subview_18 = memref.subview %arg0[0, 7, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 22>>
    %collapse_shape_19 = memref.collapse_shape %subview_18 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 22>> into memref<f32, strided<[], offset: 22>>
    %11 = memref.load %collapse_shape_19[] : memref<f32, strided<[], offset: 22>>
    %subview_20 = memref.subview %arg0[0, 7, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 21>>
    %collapse_shape_21 = memref.collapse_shape %subview_20 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 21>> into memref<f32, strided<[], offset: 21>>
    %12 = memref.load %collapse_shape_21[] : memref<f32, strided<[], offset: 21>>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<8xf32>
    linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%0 : memref<f32>) outs(%alloc : memref<8xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%alloc, %arg1 : memref<8xf32>, memref<8xf32>) outs(%alloc : memref<8xf32>) {
    ^bb0(%in: f32, %in_341: f32, %out: f32):
      %228 = arith.mulf %in, %in_341 : f32
      linalg.yield %228 : f32
    }
    %subview_22 = memref.subview %alloc[7] [1] [1] : memref<8xf32> to memref<1xf32, strided<[1], offset: 7>>
    %collapse_shape_23 = memref.collapse_shape %subview_22 [] : memref<1xf32, strided<[1], offset: 7>> into memref<f32, strided<[], offset: 7>>
    %13 = memref.load %collapse_shape_23[] : memref<f32, strided<[], offset: 7>>
    quantum.device shots(%c0_i64) ["/home/pichau/QAgents/.venv/lib/python3.12/site-packages/pennylane_lightning/liblightning_gpu_catalyst.so", "LightningGPUSimulator", "{}"]
    %14 = quantum.alloc( 8) : !quantum.reg
    %15 = quantum.extract %14[ 7] : !quantum.reg -> !quantum.bit
    %16 = arith.extf %13 : f32 to f64
    %out_qubits = quantum.custom "RY"(%16) %15 : !quantum.bit
    %17 = arith.extf %12 : f32 to f64
    %out_qubits_24 = quantum.custom "RZ"(%17) %out_qubits : !quantum.bit
    %18 = arith.extf %11 : f32 to f64
    %out_qubits_25 = quantum.custom "RY"(%18) %out_qubits_24 : !quantum.bit
    %19 = arith.extf %10 : f32 to f64
    %out_qubits_26 = quantum.custom "RZ"(%19) %out_qubits_25 : !quantum.bit
    %subview_27 = memref.subview %arg0[0, 6, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 20>>
    %collapse_shape_28 = memref.collapse_shape %subview_27 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 20>> into memref<f32, strided<[], offset: 20>>
    %20 = memref.load %collapse_shape_28[] : memref<f32, strided<[], offset: 20>>
    %subview_29 = memref.subview %arg0[0, 6, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 19>>
    %collapse_shape_30 = memref.collapse_shape %subview_29 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 19>> into memref<f32, strided<[], offset: 19>>
    %21 = memref.load %collapse_shape_30[] : memref<f32, strided<[], offset: 19>>
    %subview_31 = memref.subview %arg0[0, 6, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 18>>
    %collapse_shape_32 = memref.collapse_shape %subview_31 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 18>> into memref<f32, strided<[], offset: 18>>
    %22 = memref.load %collapse_shape_32[] : memref<f32, strided<[], offset: 18>>
    %subview_33 = memref.subview %alloc[6] [1] [1] : memref<8xf32> to memref<1xf32, strided<[1], offset: 6>>
    %collapse_shape_34 = memref.collapse_shape %subview_33 [] : memref<1xf32, strided<[1], offset: 6>> into memref<f32, strided<[], offset: 6>>
    %23 = memref.load %collapse_shape_34[] : memref<f32, strided<[], offset: 6>>
    %24 = quantum.extract %14[ 6] : !quantum.reg -> !quantum.bit
    %25 = arith.extf %23 : f32 to f64
    %out_qubits_35 = quantum.custom "RY"(%25) %24 : !quantum.bit
    %26 = arith.extf %22 : f32 to f64
    %out_qubits_36 = quantum.custom "RZ"(%26) %out_qubits_35 : !quantum.bit
    %27 = arith.extf %21 : f32 to f64
    %out_qubits_37 = quantum.custom "RY"(%27) %out_qubits_36 : !quantum.bit
    %28 = arith.extf %20 : f32 to f64
    %out_qubits_38 = quantum.custom "RZ"(%28) %out_qubits_37 : !quantum.bit
    %subview_39 = memref.subview %arg0[0, 5, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 17>>
    %collapse_shape_40 = memref.collapse_shape %subview_39 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 17>> into memref<f32, strided<[], offset: 17>>
    %29 = memref.load %collapse_shape_40[] : memref<f32, strided<[], offset: 17>>
    %subview_41 = memref.subview %arg0[0, 5, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 16>>
    %collapse_shape_42 = memref.collapse_shape %subview_41 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 16>> into memref<f32, strided<[], offset: 16>>
    %30 = memref.load %collapse_shape_42[] : memref<f32, strided<[], offset: 16>>
    %subview_43 = memref.subview %arg0[0, 5, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 15>>
    %collapse_shape_44 = memref.collapse_shape %subview_43 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 15>> into memref<f32, strided<[], offset: 15>>
    %31 = memref.load %collapse_shape_44[] : memref<f32, strided<[], offset: 15>>
    %subview_45 = memref.subview %alloc[5] [1] [1] : memref<8xf32> to memref<1xf32, strided<[1], offset: 5>>
    %collapse_shape_46 = memref.collapse_shape %subview_45 [] : memref<1xf32, strided<[1], offset: 5>> into memref<f32, strided<[], offset: 5>>
    %32 = memref.load %collapse_shape_46[] : memref<f32, strided<[], offset: 5>>
    %33 = quantum.extract %14[ 5] : !quantum.reg -> !quantum.bit
    %34 = arith.extf %32 : f32 to f64
    %out_qubits_47 = quantum.custom "RY"(%34) %33 : !quantum.bit
    %35 = arith.extf %31 : f32 to f64
    %out_qubits_48 = quantum.custom "RZ"(%35) %out_qubits_47 : !quantum.bit
    %36 = arith.extf %30 : f32 to f64
    %out_qubits_49 = quantum.custom "RY"(%36) %out_qubits_48 : !quantum.bit
    %37 = arith.extf %29 : f32 to f64
    %out_qubits_50 = quantum.custom "RZ"(%37) %out_qubits_49 : !quantum.bit
    %subview_51 = memref.subview %arg0[0, 4, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 14>>
    %collapse_shape_52 = memref.collapse_shape %subview_51 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 14>> into memref<f32, strided<[], offset: 14>>
    %38 = memref.load %collapse_shape_52[] : memref<f32, strided<[], offset: 14>>
    %subview_53 = memref.subview %arg0[0, 4, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 13>>
    %collapse_shape_54 = memref.collapse_shape %subview_53 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 13>> into memref<f32, strided<[], offset: 13>>
    %39 = memref.load %collapse_shape_54[] : memref<f32, strided<[], offset: 13>>
    %subview_55 = memref.subview %arg0[0, 4, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 12>>
    %collapse_shape_56 = memref.collapse_shape %subview_55 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 12>> into memref<f32, strided<[], offset: 12>>
    %40 = memref.load %collapse_shape_56[] : memref<f32, strided<[], offset: 12>>
    %subview_57 = memref.subview %alloc[4] [1] [1] : memref<8xf32> to memref<1xf32, strided<[1], offset: 4>>
    %collapse_shape_58 = memref.collapse_shape %subview_57 [] : memref<1xf32, strided<[1], offset: 4>> into memref<f32, strided<[], offset: 4>>
    %41 = memref.load %collapse_shape_58[] : memref<f32, strided<[], offset: 4>>
    %42 = quantum.extract %14[ 4] : !quantum.reg -> !quantum.bit
    %43 = arith.extf %41 : f32 to f64
    %out_qubits_59 = quantum.custom "RY"(%43) %42 : !quantum.bit
    %44 = arith.extf %40 : f32 to f64
    %out_qubits_60 = quantum.custom "RZ"(%44) %out_qubits_59 : !quantum.bit
    %45 = arith.extf %39 : f32 to f64
    %out_qubits_61 = quantum.custom "RY"(%45) %out_qubits_60 : !quantum.bit
    %46 = arith.extf %38 : f32 to f64
    %out_qubits_62 = quantum.custom "RZ"(%46) %out_qubits_61 : !quantum.bit
    %subview_63 = memref.subview %arg0[0, 3, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 11>>
    %collapse_shape_64 = memref.collapse_shape %subview_63 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 11>> into memref<f32, strided<[], offset: 11>>
    %47 = memref.load %collapse_shape_64[] : memref<f32, strided<[], offset: 11>>
    %subview_65 = memref.subview %arg0[0, 3, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 10>>
    %collapse_shape_66 = memref.collapse_shape %subview_65 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 10>> into memref<f32, strided<[], offset: 10>>
    %48 = memref.load %collapse_shape_66[] : memref<f32, strided<[], offset: 10>>
    %subview_67 = memref.subview %arg0[0, 3, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 9>>
    %collapse_shape_68 = memref.collapse_shape %subview_67 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 9>> into memref<f32, strided<[], offset: 9>>
    %49 = memref.load %collapse_shape_68[] : memref<f32, strided<[], offset: 9>>
    %subview_69 = memref.subview %alloc[3] [1] [1] : memref<8xf32> to memref<1xf32, strided<[1], offset: 3>>
    %collapse_shape_70 = memref.collapse_shape %subview_69 [] : memref<1xf32, strided<[1], offset: 3>> into memref<f32, strided<[], offset: 3>>
    %50 = memref.load %collapse_shape_70[] : memref<f32, strided<[], offset: 3>>
    %51 = quantum.extract %14[ 3] : !quantum.reg -> !quantum.bit
    %52 = arith.extf %50 : f32 to f64
    %out_qubits_71 = quantum.custom "RY"(%52) %51 : !quantum.bit
    %53 = arith.extf %49 : f32 to f64
    %out_qubits_72 = quantum.custom "RZ"(%53) %out_qubits_71 : !quantum.bit
    %54 = arith.extf %48 : f32 to f64
    %out_qubits_73 = quantum.custom "RY"(%54) %out_qubits_72 : !quantum.bit
    %55 = arith.extf %47 : f32 to f64
    %out_qubits_74 = quantum.custom "RZ"(%55) %out_qubits_73 : !quantum.bit
    %subview_75 = memref.subview %arg0[0, 2, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 8>>
    %collapse_shape_76 = memref.collapse_shape %subview_75 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 8>> into memref<f32, strided<[], offset: 8>>
    %56 = memref.load %collapse_shape_76[] : memref<f32, strided<[], offset: 8>>
    %subview_77 = memref.subview %arg0[0, 2, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 7>>
    %collapse_shape_78 = memref.collapse_shape %subview_77 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 7>> into memref<f32, strided<[], offset: 7>>
    %57 = memref.load %collapse_shape_78[] : memref<f32, strided<[], offset: 7>>
    %subview_79 = memref.subview %arg0[0, 2, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 6>>
    %collapse_shape_80 = memref.collapse_shape %subview_79 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 6>> into memref<f32, strided<[], offset: 6>>
    %58 = memref.load %collapse_shape_80[] : memref<f32, strided<[], offset: 6>>
    %subview_81 = memref.subview %alloc[2] [1] [1] : memref<8xf32> to memref<1xf32, strided<[1], offset: 2>>
    %collapse_shape_82 = memref.collapse_shape %subview_81 [] : memref<1xf32, strided<[1], offset: 2>> into memref<f32, strided<[], offset: 2>>
    %59 = memref.load %collapse_shape_82[] : memref<f32, strided<[], offset: 2>>
    %60 = quantum.extract %14[ 2] : !quantum.reg -> !quantum.bit
    %61 = arith.extf %59 : f32 to f64
    %out_qubits_83 = quantum.custom "RY"(%61) %60 : !quantum.bit
    %62 = arith.extf %58 : f32 to f64
    %out_qubits_84 = quantum.custom "RZ"(%62) %out_qubits_83 : !quantum.bit
    %63 = arith.extf %57 : f32 to f64
    %out_qubits_85 = quantum.custom "RY"(%63) %out_qubits_84 : !quantum.bit
    %64 = arith.extf %56 : f32 to f64
    %out_qubits_86 = quantum.custom "RZ"(%64) %out_qubits_85 : !quantum.bit
    %subview_87 = memref.subview %arg0[0, 0, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 2>>
    %collapse_shape_88 = memref.collapse_shape %subview_87 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 2>> into memref<f32, strided<[], offset: 2>>
    %65 = memref.load %collapse_shape_88[] : memref<f32, strided<[], offset: 2>>
    %subview_89 = memref.subview %arg0[0, 0, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 1>>
    %collapse_shape_90 = memref.collapse_shape %subview_89 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 1>> into memref<f32, strided<[], offset: 1>>
    %66 = memref.load %collapse_shape_90[] : memref<f32, strided<[], offset: 1>>
    %subview_91 = memref.subview %arg0[0, 0, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1]>>
    %collapse_shape_92 = memref.collapse_shape %subview_91 [] : memref<1x1x1xf32, strided<[24, 3, 1]>> into memref<f32, strided<[]>>
    %67 = memref.load %collapse_shape_92[] : memref<f32, strided<[]>>
    %subview_93 = memref.subview %alloc[0] [1] [1] : memref<8xf32> to memref<1xf32, strided<[1]>>
    %collapse_shape_94 = memref.collapse_shape %subview_93 [] : memref<1xf32, strided<[1]>> into memref<f32>
    %68 = memref.load %collapse_shape_94[] : memref<f32>
    %69 = quantum.extract %14[ 0] : !quantum.reg -> !quantum.bit
    %70 = arith.extf %68 : f32 to f64
    %out_qubits_95 = quantum.custom "RY"(%70) %69 : !quantum.bit
    %71 = arith.extf %67 : f32 to f64
    %out_qubits_96 = quantum.custom "RZ"(%71) %out_qubits_95 : !quantum.bit
    %72 = arith.extf %66 : f32 to f64
    %out_qubits_97 = quantum.custom "RY"(%72) %out_qubits_96 : !quantum.bit
    %73 = arith.extf %65 : f32 to f64
    %out_qubits_98 = quantum.custom "RZ"(%73) %out_qubits_97 : !quantum.bit
    %subview_99 = memref.subview %arg0[0, 1, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 5>>
    %collapse_shape_100 = memref.collapse_shape %subview_99 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 5>> into memref<f32, strided<[], offset: 5>>
    %74 = memref.load %collapse_shape_100[] : memref<f32, strided<[], offset: 5>>
    %subview_101 = memref.subview %arg0[0, 1, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 4>>
    %collapse_shape_102 = memref.collapse_shape %subview_101 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 4>> into memref<f32, strided<[], offset: 4>>
    %75 = memref.load %collapse_shape_102[] : memref<f32, strided<[], offset: 4>>
    %subview_103 = memref.subview %arg0[0, 1, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 3>>
    %collapse_shape_104 = memref.collapse_shape %subview_103 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 3>> into memref<f32, strided<[], offset: 3>>
    %76 = memref.load %collapse_shape_104[] : memref<f32, strided<[], offset: 3>>
    %subview_105 = memref.subview %alloc[1] [1] [1] : memref<8xf32> to memref<1xf32, strided<[1], offset: 1>>
    %collapse_shape_106 = memref.collapse_shape %subview_105 [] : memref<1xf32, strided<[1], offset: 1>> into memref<f32, strided<[], offset: 1>>
    %77 = memref.load %collapse_shape_106[] : memref<f32, strided<[], offset: 1>>
    memref.dealloc %alloc : memref<8xf32>
    %78 = quantum.extract %14[ 1] : !quantum.reg -> !quantum.bit
    %79 = arith.extf %77 : f32 to f64
    %out_qubits_107 = quantum.custom "RY"(%79) %78 : !quantum.bit
    %80 = arith.extf %76 : f32 to f64
    %out_qubits_108 = quantum.custom "RZ"(%80) %out_qubits_107 : !quantum.bit
    %81 = arith.extf %75 : f32 to f64
    %out_qubits_109 = quantum.custom "RY"(%81) %out_qubits_108 : !quantum.bit
    %82 = arith.extf %74 : f32 to f64
    %out_qubits_110 = quantum.custom "RZ"(%82) %out_qubits_109 : !quantum.bit
    %out_qubits_111:2 = quantum.custom "CNOT"() %out_qubits_98, %out_qubits_110 : !quantum.bit, !quantum.bit
    %out_qubits_112:2 = quantum.custom "CNOT"() %out_qubits_111#1, %out_qubits_86 : !quantum.bit, !quantum.bit
    %out_qubits_113:2 = quantum.custom "CNOT"() %out_qubits_112#1, %out_qubits_74 : !quantum.bit, !quantum.bit
    %out_qubits_114:2 = quantum.custom "CNOT"() %out_qubits_113#1, %out_qubits_62 : !quantum.bit, !quantum.bit
    %out_qubits_115:2 = quantum.custom "CNOT"() %out_qubits_114#1, %out_qubits_50 : !quantum.bit, !quantum.bit
    %out_qubits_116:2 = quantum.custom "CNOT"() %out_qubits_115#1, %out_qubits_38 : !quantum.bit, !quantum.bit
    %out_qubits_117:2 = quantum.custom "CNOT"() %out_qubits_116#1, %out_qubits_26 : !quantum.bit, !quantum.bit
    %83 = arith.extf %9 : f32 to f64
    %out_qubits_118 = quantum.custom "RZ"(%83) %out_qubits_117#0 : !quantum.bit
    %84 = arith.extf %8 : f32 to f64
    %out_qubits_119 = quantum.custom "RY"(%84) %out_qubits_118 : !quantum.bit
    %85 = arith.extf %7 : f32 to f64
    %out_qubits_120 = quantum.custom "RZ"(%85) %out_qubits_119 : !quantum.bit
    %subview_121 = memref.subview %arg0[1, 4, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 38>>
    %collapse_shape_122 = memref.collapse_shape %subview_121 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 38>> into memref<f32, strided<[], offset: 38>>
    %86 = memref.load %collapse_shape_122[] : memref<f32, strided<[], offset: 38>>
    %subview_123 = memref.subview %arg0[1, 4, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 37>>
    %collapse_shape_124 = memref.collapse_shape %subview_123 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 37>> into memref<f32, strided<[], offset: 37>>
    %87 = memref.load %collapse_shape_124[] : memref<f32, strided<[], offset: 37>>
    %subview_125 = memref.subview %arg0[1, 4, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 36>>
    %collapse_shape_126 = memref.collapse_shape %subview_125 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 36>> into memref<f32, strided<[], offset: 36>>
    %88 = memref.load %collapse_shape_126[] : memref<f32, strided<[], offset: 36>>
    %89 = arith.extf %88 : f32 to f64
    %out_qubits_127 = quantum.custom "RZ"(%89) %out_qubits_115#0 : !quantum.bit
    %90 = arith.extf %87 : f32 to f64
    %out_qubits_128 = quantum.custom "RY"(%90) %out_qubits_127 : !quantum.bit
    %91 = arith.extf %86 : f32 to f64
    %out_qubits_129 = quantum.custom "RZ"(%91) %out_qubits_128 : !quantum.bit
    %subview_130 = memref.subview %arg0[1, 0, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 26>>
    %collapse_shape_131 = memref.collapse_shape %subview_130 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 26>> into memref<f32, strided<[], offset: 26>>
    %92 = memref.load %collapse_shape_131[] : memref<f32, strided<[], offset: 26>>
    %subview_132 = memref.subview %arg0[1, 0, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 25>>
    %collapse_shape_133 = memref.collapse_shape %subview_132 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 25>> into memref<f32, strided<[], offset: 25>>
    %93 = memref.load %collapse_shape_133[] : memref<f32, strided<[], offset: 25>>
    %subview_134 = memref.subview %arg0[1, 0, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 24>>
    %collapse_shape_135 = memref.collapse_shape %subview_134 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 24>> into memref<f32, strided<[], offset: 24>>
    %94 = memref.load %collapse_shape_135[] : memref<f32, strided<[], offset: 24>>
    %out_qubits_136:2 = quantum.custom "CNOT"() %out_qubits_117#1, %out_qubits_111#0 : !quantum.bit, !quantum.bit
    %95 = arith.extf %94 : f32 to f64
    %out_qubits_137 = quantum.custom "RZ"(%95) %out_qubits_136#1 : !quantum.bit
    %96 = arith.extf %93 : f32 to f64
    %out_qubits_138 = quantum.custom "RY"(%96) %out_qubits_137 : !quantum.bit
    %97 = arith.extf %92 : f32 to f64
    %out_qubits_139 = quantum.custom "RZ"(%97) %out_qubits_138 : !quantum.bit
    %subview_140 = memref.subview %arg0[1, 2, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 32>>
    %collapse_shape_141 = memref.collapse_shape %subview_140 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 32>> into memref<f32, strided<[], offset: 32>>
    %98 = memref.load %collapse_shape_141[] : memref<f32, strided<[], offset: 32>>
    %subview_142 = memref.subview %arg0[1, 2, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 31>>
    %collapse_shape_143 = memref.collapse_shape %subview_142 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 31>> into memref<f32, strided<[], offset: 31>>
    %99 = memref.load %collapse_shape_143[] : memref<f32, strided<[], offset: 31>>
    %subview_144 = memref.subview %arg0[1, 2, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 30>>
    %collapse_shape_145 = memref.collapse_shape %subview_144 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 30>> into memref<f32, strided<[], offset: 30>>
    %100 = memref.load %collapse_shape_145[] : memref<f32, strided<[], offset: 30>>
    %101 = arith.extf %100 : f32 to f64
    %out_qubits_146 = quantum.custom "RZ"(%101) %out_qubits_113#0 : !quantum.bit
    %102 = arith.extf %99 : f32 to f64
    %out_qubits_147 = quantum.custom "RY"(%102) %out_qubits_146 : !quantum.bit
    %103 = arith.extf %98 : f32 to f64
    %out_qubits_148 = quantum.custom "RZ"(%103) %out_qubits_147 : !quantum.bit
    %out_qubits_149:2 = quantum.custom "CNOT"() %out_qubits_139, %out_qubits_148 : !quantum.bit, !quantum.bit
    %out_qubits_150:2 = quantum.custom "CNOT"() %out_qubits_149#1, %out_qubits_129 : !quantum.bit, !quantum.bit
    %out_qubits_151:2 = quantum.custom "CNOT"() %out_qubits_150#1, %out_qubits_120 : !quantum.bit, !quantum.bit
    %out_qubits_152:2 = quantum.custom "CNOT"() %out_qubits_151#1, %out_qubits_149#0 : !quantum.bit, !quantum.bit
    %104 = arith.extf %6 : f32 to f64
    %out_qubits_153 = quantum.custom "RZ"(%104) %out_qubits_152#1 : !quantum.bit
    %105 = arith.extf %5 : f32 to f64
    %out_qubits_154 = quantum.custom "RY"(%105) %out_qubits_153 : !quantum.bit
    %106 = arith.extf %4 : f32 to f64
    %out_qubits_155 = quantum.custom "RZ"(%106) %out_qubits_154 : !quantum.bit
    %subview_156 = memref.subview %arg0[2, 3, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 59>>
    %collapse_shape_157 = memref.collapse_shape %subview_156 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 59>> into memref<f32, strided<[], offset: 59>>
    %107 = memref.load %collapse_shape_157[] : memref<f32, strided<[], offset: 59>>
    %subview_158 = memref.subview %arg0[2, 3, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 58>>
    %collapse_shape_159 = memref.collapse_shape %subview_158 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 58>> into memref<f32, strided<[], offset: 58>>
    %108 = memref.load %collapse_shape_159[] : memref<f32, strided<[], offset: 58>>
    %subview_160 = memref.subview %arg0[2, 3, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 57>>
    %collapse_shape_161 = memref.collapse_shape %subview_160 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 57>> into memref<f32, strided<[], offset: 57>>
    %109 = memref.load %collapse_shape_161[] : memref<f32, strided<[], offset: 57>>
    %subview_162 = memref.subview %arg0[1, 5, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 41>>
    %collapse_shape_163 = memref.collapse_shape %subview_162 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 41>> into memref<f32, strided<[], offset: 41>>
    %110 = memref.load %collapse_shape_163[] : memref<f32, strided<[], offset: 41>>
    %subview_164 = memref.subview %arg0[1, 5, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 40>>
    %collapse_shape_165 = memref.collapse_shape %subview_164 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 40>> into memref<f32, strided<[], offset: 40>>
    %111 = memref.load %collapse_shape_165[] : memref<f32, strided<[], offset: 40>>
    %subview_166 = memref.subview %arg0[1, 5, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 39>>
    %collapse_shape_167 = memref.collapse_shape %subview_166 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 39>> into memref<f32, strided<[], offset: 39>>
    %112 = memref.load %collapse_shape_167[] : memref<f32, strided<[], offset: 39>>
    %113 = arith.extf %112 : f32 to f64
    %out_qubits_168 = quantum.custom "RZ"(%113) %out_qubits_116#0 : !quantum.bit
    %114 = arith.extf %111 : f32 to f64
    %out_qubits_169 = quantum.custom "RY"(%114) %out_qubits_168 : !quantum.bit
    %115 = arith.extf %110 : f32 to f64
    %out_qubits_170 = quantum.custom "RZ"(%115) %out_qubits_169 : !quantum.bit
    %subview_171 = memref.subview %arg0[1, 1, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 29>>
    %collapse_shape_172 = memref.collapse_shape %subview_171 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 29>> into memref<f32, strided<[], offset: 29>>
    %116 = memref.load %collapse_shape_172[] : memref<f32, strided<[], offset: 29>>
    %subview_173 = memref.subview %arg0[1, 1, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 28>>
    %collapse_shape_174 = memref.collapse_shape %subview_173 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 28>> into memref<f32, strided<[], offset: 28>>
    %117 = memref.load %collapse_shape_174[] : memref<f32, strided<[], offset: 28>>
    %subview_175 = memref.subview %arg0[1, 1, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 27>>
    %collapse_shape_176 = memref.collapse_shape %subview_175 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 27>> into memref<f32, strided<[], offset: 27>>
    %118 = memref.load %collapse_shape_176[] : memref<f32, strided<[], offset: 27>>
    %119 = arith.extf %118 : f32 to f64
    %out_qubits_177 = quantum.custom "RZ"(%119) %out_qubits_112#0 : !quantum.bit
    %120 = arith.extf %117 : f32 to f64
    %out_qubits_178 = quantum.custom "RY"(%120) %out_qubits_177 : !quantum.bit
    %121 = arith.extf %116 : f32 to f64
    %out_qubits_179 = quantum.custom "RZ"(%121) %out_qubits_178 : !quantum.bit
    %subview_180 = memref.subview %arg0[1, 3, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 35>>
    %collapse_shape_181 = memref.collapse_shape %subview_180 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 35>> into memref<f32, strided<[], offset: 35>>
    %122 = memref.load %collapse_shape_181[] : memref<f32, strided<[], offset: 35>>
    %subview_182 = memref.subview %arg0[1, 3, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 34>>
    %collapse_shape_183 = memref.collapse_shape %subview_182 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 34>> into memref<f32, strided<[], offset: 34>>
    %123 = memref.load %collapse_shape_183[] : memref<f32, strided<[], offset: 34>>
    %subview_184 = memref.subview %arg0[1, 3, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 33>>
    %collapse_shape_185 = memref.collapse_shape %subview_184 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 33>> into memref<f32, strided<[], offset: 33>>
    %124 = memref.load %collapse_shape_185[] : memref<f32, strided<[], offset: 33>>
    %125 = arith.extf %124 : f32 to f64
    %out_qubits_186 = quantum.custom "RZ"(%125) %out_qubits_114#0 : !quantum.bit
    %126 = arith.extf %123 : f32 to f64
    %out_qubits_187 = quantum.custom "RY"(%126) %out_qubits_186 : !quantum.bit
    %127 = arith.extf %122 : f32 to f64
    %out_qubits_188 = quantum.custom "RZ"(%127) %out_qubits_187 : !quantum.bit
    %out_qubits_189:2 = quantum.custom "CNOT"() %out_qubits_179, %out_qubits_188 : !quantum.bit, !quantum.bit
    %out_qubits_190:2 = quantum.custom "CNOT"() %out_qubits_189#1, %out_qubits_170 : !quantum.bit, !quantum.bit
    %128 = arith.extf %109 : f32 to f64
    %out_qubits_191 = quantum.custom "RZ"(%128) %out_qubits_190#0 : !quantum.bit
    %129 = arith.extf %108 : f32 to f64
    %out_qubits_192 = quantum.custom "RY"(%129) %out_qubits_191 : !quantum.bit
    %130 = arith.extf %107 : f32 to f64
    %out_qubits_193 = quantum.custom "RZ"(%130) %out_qubits_192 : !quantum.bit
    %out_qubits_194:2 = quantum.custom "CNOT"() %out_qubits_155, %out_qubits_193 : !quantum.bit, !quantum.bit
    %subview_195 = memref.subview %arg0[2, 2, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 56>>
    %collapse_shape_196 = memref.collapse_shape %subview_195 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 56>> into memref<f32, strided<[], offset: 56>>
    %131 = memref.load %collapse_shape_196[] : memref<f32, strided<[], offset: 56>>
    %subview_197 = memref.subview %arg0[2, 2, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 55>>
    %collapse_shape_198 = memref.collapse_shape %subview_197 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 55>> into memref<f32, strided<[], offset: 55>>
    %132 = memref.load %collapse_shape_198[] : memref<f32, strided<[], offset: 55>>
    %subview_199 = memref.subview %arg0[2, 2, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 54>>
    %collapse_shape_200 = memref.collapse_shape %subview_199 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 54>> into memref<f32, strided<[], offset: 54>>
    %133 = memref.load %collapse_shape_200[] : memref<f32, strided<[], offset: 54>>
    %134 = arith.extf %133 : f32 to f64
    %out_qubits_201 = quantum.custom "RZ"(%134) %out_qubits_150#0 : !quantum.bit
    %135 = arith.extf %132 : f32 to f64
    %out_qubits_202 = quantum.custom "RY"(%135) %out_qubits_201 : !quantum.bit
    %136 = arith.extf %131 : f32 to f64
    %out_qubits_203 = quantum.custom "RZ"(%136) %out_qubits_202 : !quantum.bit
    %subview_204 = memref.subview %arg0[2, 5, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 65>>
    %collapse_shape_205 = memref.collapse_shape %subview_204 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 65>> into memref<f32, strided<[], offset: 65>>
    %137 = memref.load %collapse_shape_205[] : memref<f32, strided<[], offset: 65>>
    %subview_206 = memref.subview %arg0[2, 5, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 64>>
    %collapse_shape_207 = memref.collapse_shape %subview_206 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 64>> into memref<f32, strided<[], offset: 64>>
    %138 = memref.load %collapse_shape_207[] : memref<f32, strided<[], offset: 64>>
    %subview_208 = memref.subview %arg0[2, 5, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 63>>
    %collapse_shape_209 = memref.collapse_shape %subview_208 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 63>> into memref<f32, strided<[], offset: 63>>
    %139 = memref.load %collapse_shape_209[] : memref<f32, strided<[], offset: 63>>
    %subview_210 = memref.subview %arg0[1, 7, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 47>>
    %collapse_shape_211 = memref.collapse_shape %subview_210 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 47>> into memref<f32, strided<[], offset: 47>>
    %140 = memref.load %collapse_shape_211[] : memref<f32, strided<[], offset: 47>>
    %subview_212 = memref.subview %arg0[1, 7, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 46>>
    %collapse_shape_213 = memref.collapse_shape %subview_212 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 46>> into memref<f32, strided<[], offset: 46>>
    %141 = memref.load %collapse_shape_213[] : memref<f32, strided<[], offset: 46>>
    %subview_214 = memref.subview %arg0[1, 7, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 45>>
    %collapse_shape_215 = memref.collapse_shape %subview_214 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 45>> into memref<f32, strided<[], offset: 45>>
    %142 = memref.load %collapse_shape_215[] : memref<f32, strided<[], offset: 45>>
    %143 = arith.extf %142 : f32 to f64
    %out_qubits_216 = quantum.custom "RZ"(%143) %out_qubits_136#0 : !quantum.bit
    %144 = arith.extf %141 : f32 to f64
    %out_qubits_217 = quantum.custom "RY"(%144) %out_qubits_216 : !quantum.bit
    %145 = arith.extf %140 : f32 to f64
    %out_qubits_218 = quantum.custom "RZ"(%145) %out_qubits_217 : !quantum.bit
    %out_qubits_219:2 = quantum.custom "CNOT"() %out_qubits_190#1, %out_qubits_218 : !quantum.bit, !quantum.bit
    %146 = arith.extf %139 : f32 to f64
    %out_qubits_220 = quantum.custom "RZ"(%146) %out_qubits_219#0 : !quantum.bit
    %147 = arith.extf %138 : f32 to f64
    %out_qubits_221 = quantum.custom "RY"(%147) %out_qubits_220 : !quantum.bit
    %148 = arith.extf %137 : f32 to f64
    %out_qubits_222 = quantum.custom "RZ"(%148) %out_qubits_221 : !quantum.bit
    %out_qubits_223:2 = quantum.custom "CNOT"() %out_qubits_203, %out_qubits_222 : !quantum.bit, !quantum.bit
    %out_qubits_224:2 = quantum.custom "CNOT"() %out_qubits_223#1, %out_qubits_194#0 : !quantum.bit, !quantum.bit
    %149 = arith.extf %3 : f32 to f64
    %out_qubits_225 = quantum.custom "RZ"(%149) %out_qubits_224#1 : !quantum.bit
    %150 = arith.extf %2 : f32 to f64
    %out_qubits_226 = quantum.custom "RY"(%150) %out_qubits_225 : !quantum.bit
    %151 = arith.extf %1 : f32 to f64
    %out_qubits_227 = quantum.custom "RZ"(%151) %out_qubits_226 : !quantum.bit
    %subview_228 = memref.subview %arg0[3, 4, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 86>>
    %collapse_shape_229 = memref.collapse_shape %subview_228 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 86>> into memref<f32, strided<[], offset: 86>>
    %152 = memref.load %collapse_shape_229[] : memref<f32, strided<[], offset: 86>>
    %subview_230 = memref.subview %arg0[3, 4, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 85>>
    %collapse_shape_231 = memref.collapse_shape %subview_230 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 85>> into memref<f32, strided<[], offset: 85>>
    %153 = memref.load %collapse_shape_231[] : memref<f32, strided<[], offset: 85>>
    %subview_232 = memref.subview %arg0[3, 4, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 84>>
    %collapse_shape_233 = memref.collapse_shape %subview_232 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 84>> into memref<f32, strided<[], offset: 84>>
    %154 = memref.load %collapse_shape_233[] : memref<f32, strided<[], offset: 84>>
    %subview_234 = memref.subview %arg0[2, 7, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 71>>
    %collapse_shape_235 = memref.collapse_shape %subview_234 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 71>> into memref<f32, strided<[], offset: 71>>
    %155 = memref.load %collapse_shape_235[] : memref<f32, strided<[], offset: 71>>
    %subview_236 = memref.subview %arg0[2, 7, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 70>>
    %collapse_shape_237 = memref.collapse_shape %subview_236 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 70>> into memref<f32, strided<[], offset: 70>>
    %156 = memref.load %collapse_shape_237[] : memref<f32, strided<[], offset: 70>>
    %subview_238 = memref.subview %arg0[2, 7, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 69>>
    %collapse_shape_239 = memref.collapse_shape %subview_238 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 69>> into memref<f32, strided<[], offset: 69>>
    %157 = memref.load %collapse_shape_239[] : memref<f32, strided<[], offset: 69>>
    %out_qubits_240:2 = quantum.custom "CNOT"() %out_qubits_219#1, %out_qubits_189#0 : !quantum.bit, !quantum.bit
    %158 = arith.extf %157 : f32 to f64
    %out_qubits_241 = quantum.custom "RZ"(%158) %out_qubits_240#0 : !quantum.bit
    %159 = arith.extf %156 : f32 to f64
    %out_qubits_242 = quantum.custom "RY"(%159) %out_qubits_241 : !quantum.bit
    %160 = arith.extf %155 : f32 to f64
    %out_qubits_243 = quantum.custom "RZ"(%160) %out_qubits_242 : !quantum.bit
    %subview_244 = memref.subview %arg0[2, 1, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 53>>
    %collapse_shape_245 = memref.collapse_shape %subview_244 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 53>> into memref<f32, strided<[], offset: 53>>
    %161 = memref.load %collapse_shape_245[] : memref<f32, strided<[], offset: 53>>
    %subview_246 = memref.subview %arg0[2, 1, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 52>>
    %collapse_shape_247 = memref.collapse_shape %subview_246 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 52>> into memref<f32, strided<[], offset: 52>>
    %162 = memref.load %collapse_shape_247[] : memref<f32, strided<[], offset: 52>>
    %subview_248 = memref.subview %arg0[2, 1, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 51>>
    %collapse_shape_249 = memref.collapse_shape %subview_248 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 51>> into memref<f32, strided<[], offset: 51>>
    %163 = memref.load %collapse_shape_249[] : memref<f32, strided<[], offset: 51>>
    %164 = arith.extf %163 : f32 to f64
    %out_qubits_250 = quantum.custom "RZ"(%164) %out_qubits_240#1 : !quantum.bit
    %165 = arith.extf %162 : f32 to f64
    %out_qubits_251 = quantum.custom "RY"(%165) %out_qubits_250 : !quantum.bit
    %166 = arith.extf %161 : f32 to f64
    %out_qubits_252 = quantum.custom "RZ"(%166) %out_qubits_251 : !quantum.bit
    %subview_253 = memref.subview %arg0[2, 4, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 62>>
    %collapse_shape_254 = memref.collapse_shape %subview_253 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 62>> into memref<f32, strided<[], offset: 62>>
    %167 = memref.load %collapse_shape_254[] : memref<f32, strided<[], offset: 62>>
    %subview_255 = memref.subview %arg0[2, 4, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 61>>
    %collapse_shape_256 = memref.collapse_shape %subview_255 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 61>> into memref<f32, strided<[], offset: 61>>
    %168 = memref.load %collapse_shape_256[] : memref<f32, strided<[], offset: 61>>
    %subview_257 = memref.subview %arg0[2, 4, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 60>>
    %collapse_shape_258 = memref.collapse_shape %subview_257 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 60>> into memref<f32, strided<[], offset: 60>>
    %169 = memref.load %collapse_shape_258[] : memref<f32, strided<[], offset: 60>>
    %170 = arith.extf %169 : f32 to f64
    %out_qubits_259 = quantum.custom "RZ"(%170) %out_qubits_151#0 : !quantum.bit
    %171 = arith.extf %168 : f32 to f64
    %out_qubits_260 = quantum.custom "RY"(%171) %out_qubits_259 : !quantum.bit
    %172 = arith.extf %167 : f32 to f64
    %out_qubits_261 = quantum.custom "RZ"(%172) %out_qubits_260 : !quantum.bit
    %out_qubits_262:2 = quantum.custom "CNOT"() %out_qubits_252, %out_qubits_261 : !quantum.bit, !quantum.bit
    %out_qubits_263:2 = quantum.custom "CNOT"() %out_qubits_262#1, %out_qubits_243 : !quantum.bit, !quantum.bit
    %173 = arith.extf %154 : f32 to f64
    %out_qubits_264 = quantum.custom "RZ"(%173) %out_qubits_263#0 : !quantum.bit
    %174 = arith.extf %153 : f32 to f64
    %out_qubits_265 = quantum.custom "RY"(%174) %out_qubits_264 : !quantum.bit
    %175 = arith.extf %152 : f32 to f64
    %out_qubits_266 = quantum.custom "RZ"(%175) %out_qubits_265 : !quantum.bit
    %out_qubits_267:2 = quantum.custom "CNOT"() %out_qubits_227, %out_qubits_266 : !quantum.bit, !quantum.bit
    %out_qubits_268:2 = quantum.custom "CNOT"() %out_qubits_267#1, %out_qubits_267#0 : !quantum.bit, !quantum.bit
    %subview_269 = memref.subview %arg0[3, 1, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 77>>
    %collapse_shape_270 = memref.collapse_shape %subview_269 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 77>> into memref<f32, strided<[], offset: 77>>
    %176 = memref.load %collapse_shape_270[] : memref<f32, strided<[], offset: 77>>
    %subview_271 = memref.subview %arg0[3, 1, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 76>>
    %collapse_shape_272 = memref.collapse_shape %subview_271 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 76>> into memref<f32, strided<[], offset: 76>>
    %177 = memref.load %collapse_shape_272[] : memref<f32, strided<[], offset: 76>>
    %subview_273 = memref.subview %arg0[3, 1, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 75>>
    %collapse_shape_274 = memref.collapse_shape %subview_273 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 75>> into memref<f32, strided<[], offset: 75>>
    %178 = memref.load %collapse_shape_274[] : memref<f32, strided<[], offset: 75>>
    %subview_275 = memref.subview %arg0[2, 6, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 68>>
    %collapse_shape_276 = memref.collapse_shape %subview_275 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 68>> into memref<f32, strided<[], offset: 68>>
    %179 = memref.load %collapse_shape_276[] : memref<f32, strided<[], offset: 68>>
    %subview_277 = memref.subview %arg0[2, 6, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 67>>
    %collapse_shape_278 = memref.collapse_shape %subview_277 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 67>> into memref<f32, strided<[], offset: 67>>
    %180 = memref.load %collapse_shape_278[] : memref<f32, strided<[], offset: 67>>
    %subview_279 = memref.subview %arg0[2, 6, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 66>>
    %collapse_shape_280 = memref.collapse_shape %subview_279 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 66>> into memref<f32, strided<[], offset: 66>>
    %181 = memref.load %collapse_shape_280[] : memref<f32, strided<[], offset: 66>>
    %182 = arith.extf %181 : f32 to f64
    %out_qubits_281 = quantum.custom "RZ"(%182) %out_qubits_152#0 : !quantum.bit
    %183 = arith.extf %180 : f32 to f64
    %out_qubits_282 = quantum.custom "RY"(%183) %out_qubits_281 : !quantum.bit
    %184 = arith.extf %179 : f32 to f64
    %out_qubits_283 = quantum.custom "RZ"(%184) %out_qubits_282 : !quantum.bit
    %out_qubits_284:2 = quantum.custom "CNOT"() %out_qubits_194#1, %out_qubits_283 : !quantum.bit, !quantum.bit
    %out_qubits_285:2 = quantum.custom "CNOT"() %out_qubits_284#1, %out_qubits_262#0 : !quantum.bit, !quantum.bit
    %185 = arith.extf %178 : f32 to f64
    %out_qubits_286 = quantum.custom "RZ"(%185) %out_qubits_285#1 : !quantum.bit
    %186 = arith.extf %177 : f32 to f64
    %out_qubits_287 = quantum.custom "RY"(%186) %out_qubits_286 : !quantum.bit
    %187 = arith.extf %176 : f32 to f64
    %out_qubits_288 = quantum.custom "RZ"(%187) %out_qubits_287 : !quantum.bit
    %subview_289 = memref.subview %arg0[3, 5, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 89>>
    %collapse_shape_290 = memref.collapse_shape %subview_289 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 89>> into memref<f32, strided<[], offset: 89>>
    %188 = memref.load %collapse_shape_290[] : memref<f32, strided<[], offset: 89>>
    %subview_291 = memref.subview %arg0[3, 5, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 88>>
    %collapse_shape_292 = memref.collapse_shape %subview_291 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 88>> into memref<f32, strided<[], offset: 88>>
    %189 = memref.load %collapse_shape_292[] : memref<f32, strided<[], offset: 88>>
    %subview_293 = memref.subview %arg0[3, 5, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 87>>
    %collapse_shape_294 = memref.collapse_shape %subview_293 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 87>> into memref<f32, strided<[], offset: 87>>
    %190 = memref.load %collapse_shape_294[] : memref<f32, strided<[], offset: 87>>
    %191 = arith.extf %190 : f32 to f64
    %out_qubits_295 = quantum.custom "RZ"(%191) %out_qubits_224#0 : !quantum.bit
    %192 = arith.extf %189 : f32 to f64
    %out_qubits_296 = quantum.custom "RY"(%192) %out_qubits_295 : !quantum.bit
    %193 = arith.extf %188 : f32 to f64
    %out_qubits_297 = quantum.custom "RZ"(%193) %out_qubits_296 : !quantum.bit
    %out_qubits_298:2 = quantum.custom "CNOT"() %out_qubits_288, %out_qubits_297 : !quantum.bit, !quantum.bit
    %out_qubits_299:2 = quantum.custom "CNOT"() %out_qubits_298#1, %out_qubits_298#0 : !quantum.bit, !quantum.bit
    %subview_300 = memref.subview %arg0[3, 2, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 80>>
    %collapse_shape_301 = memref.collapse_shape %subview_300 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 80>> into memref<f32, strided<[], offset: 80>>
    %194 = memref.load %collapse_shape_301[] : memref<f32, strided<[], offset: 80>>
    %subview_302 = memref.subview %arg0[3, 2, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 79>>
    %collapse_shape_303 = memref.collapse_shape %subview_302 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 79>> into memref<f32, strided<[], offset: 79>>
    %195 = memref.load %collapse_shape_303[] : memref<f32, strided<[], offset: 79>>
    %subview_304 = memref.subview %arg0[3, 2, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 78>>
    %collapse_shape_305 = memref.collapse_shape %subview_304 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 78>> into memref<f32, strided<[], offset: 78>>
    %196 = memref.load %collapse_shape_305[] : memref<f32, strided<[], offset: 78>>
    %out_qubits_306:2 = quantum.custom "CNOT"() %out_qubits_263#1, %out_qubits_223#0 : !quantum.bit, !quantum.bit
    %197 = arith.extf %196 : f32 to f64
    %out_qubits_307 = quantum.custom "RZ"(%197) %out_qubits_306#1 : !quantum.bit
    %198 = arith.extf %195 : f32 to f64
    %out_qubits_308 = quantum.custom "RY"(%198) %out_qubits_307 : !quantum.bit
    %199 = arith.extf %194 : f32 to f64
    %out_qubits_309 = quantum.custom "RZ"(%199) %out_qubits_308 : !quantum.bit
    %subview_310 = memref.subview %arg0[3, 6, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 92>>
    %collapse_shape_311 = memref.collapse_shape %subview_310 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 92>> into memref<f32, strided<[], offset: 92>>
    %200 = memref.load %collapse_shape_311[] : memref<f32, strided<[], offset: 92>>
    %subview_312 = memref.subview %arg0[3, 6, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 91>>
    %collapse_shape_313 = memref.collapse_shape %subview_312 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 91>> into memref<f32, strided<[], offset: 91>>
    %201 = memref.load %collapse_shape_313[] : memref<f32, strided<[], offset: 91>>
    %subview_314 = memref.subview %arg0[3, 6, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 90>>
    %collapse_shape_315 = memref.collapse_shape %subview_314 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 90>> into memref<f32, strided<[], offset: 90>>
    %202 = memref.load %collapse_shape_315[] : memref<f32, strided<[], offset: 90>>
    %203 = arith.extf %202 : f32 to f64
    %out_qubits_316 = quantum.custom "RZ"(%203) %out_qubits_285#0 : !quantum.bit
    %204 = arith.extf %201 : f32 to f64
    %out_qubits_317 = quantum.custom "RY"(%204) %out_qubits_316 : !quantum.bit
    %205 = arith.extf %200 : f32 to f64
    %out_qubits_318 = quantum.custom "RZ"(%205) %out_qubits_317 : !quantum.bit
    %out_qubits_319:2 = quantum.custom "CNOT"() %out_qubits_309, %out_qubits_318 : !quantum.bit, !quantum.bit
    %out_qubits_320:2 = quantum.custom "CNOT"() %out_qubits_319#1, %out_qubits_319#0 : !quantum.bit, !quantum.bit
    %subview_321 = memref.subview %arg0[3, 3, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 83>>
    %collapse_shape_322 = memref.collapse_shape %subview_321 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 83>> into memref<f32, strided<[], offset: 83>>
    %206 = memref.load %collapse_shape_322[] : memref<f32, strided<[], offset: 83>>
    %subview_323 = memref.subview %arg0[3, 3, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 82>>
    %collapse_shape_324 = memref.collapse_shape %subview_323 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 82>> into memref<f32, strided<[], offset: 82>>
    %207 = memref.load %collapse_shape_324[] : memref<f32, strided<[], offset: 82>>
    %subview_325 = memref.subview %arg0[3, 3, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 81>>
    %collapse_shape_326 = memref.collapse_shape %subview_325 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 81>> into memref<f32, strided<[], offset: 81>>
    %208 = memref.load %collapse_shape_326[] : memref<f32, strided<[], offset: 81>>
    %209 = arith.extf %208 : f32 to f64
    %out_qubits_327 = quantum.custom "RZ"(%209) %out_qubits_284#0 : !quantum.bit
    %210 = arith.extf %207 : f32 to f64
    %out_qubits_328 = quantum.custom "RY"(%210) %out_qubits_327 : !quantum.bit
    %211 = arith.extf %206 : f32 to f64
    %out_qubits_329 = quantum.custom "RZ"(%211) %out_qubits_328 : !quantum.bit
    %subview_330 = memref.subview %arg0[3, 7, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 95>>
    %collapse_shape_331 = memref.collapse_shape %subview_330 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 95>> into memref<f32, strided<[], offset: 95>>
    %212 = memref.load %collapse_shape_331[] : memref<f32, strided<[], offset: 95>>
    %subview_332 = memref.subview %arg0[3, 7, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 94>>
    %collapse_shape_333 = memref.collapse_shape %subview_332 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 94>> into memref<f32, strided<[], offset: 94>>
    %213 = memref.load %collapse_shape_333[] : memref<f32, strided<[], offset: 94>>
    %subview_334 = memref.subview %arg0[3, 7, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 93>>
    %collapse_shape_335 = memref.collapse_shape %subview_334 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 93>> into memref<f32, strided<[], offset: 93>>
    %214 = memref.load %collapse_shape_335[] : memref<f32, strided<[], offset: 93>>
    %215 = arith.extf %214 : f32 to f64
    %out_qubits_336 = quantum.custom "RZ"(%215) %out_qubits_306#0 : !quantum.bit
    %216 = arith.extf %213 : f32 to f64
    %out_qubits_337 = quantum.custom "RY"(%216) %out_qubits_336 : !quantum.bit
    %217 = arith.extf %212 : f32 to f64
    %out_qubits_338 = quantum.custom "RZ"(%217) %out_qubits_337 : !quantum.bit
    %out_qubits_339:2 = quantum.custom "CNOT"() %out_qubits_329, %out_qubits_338 : !quantum.bit, !quantum.bit
    %out_qubits_340:2 = quantum.custom "CNOT"() %out_qubits_339#1, %out_qubits_339#0 : !quantum.bit, !quantum.bit
    %218 = quantum.namedobs %out_qubits_268#1[ PauliZ] : !quantum.obs
    %219 = quantum.expval %218 : f64
    %220 = quantum.insert %14[ 0], %out_qubits_268#1 : !quantum.reg, !quantum.bit
    %221 = quantum.insert %220[ 1], %out_qubits_299#1 : !quantum.reg, !quantum.bit
    %222 = quantum.insert %221[ 2], %out_qubits_320#1 : !quantum.reg, !quantum.bit
    %223 = quantum.insert %222[ 3], %out_qubits_340#1 : !quantum.reg, !quantum.bit
    %224 = quantum.insert %223[ 4], %out_qubits_268#0 : !quantum.reg, !quantum.bit
    %225 = quantum.insert %224[ 5], %out_qubits_299#0 : !quantum.reg, !quantum.bit
    %226 = quantum.insert %225[ 6], %out_qubits_320#0 : !quantum.reg, !quantum.bit
    %227 = quantum.insert %226[ 7], %out_qubits_340#0 : !quantum.reg, !quantum.bit
    return %227, %219 : !quantum.reg, f64
  }
  func.func private @qnode_forward_0.pcount(%arg0: memref<4x8x3xf32>, %arg1: memref<8xf32>) -> index {
    %idx1 = index.constant 1
    %idx0 = index.constant 0
    %alloca = memref.alloca() : memref<index>
    memref.store %idx0, %alloca[] : memref<index>
    %0 = memref.load %alloca[] : memref<index>
    %1 = index.add %0, %idx1
    memref.store %1, %alloca[] : memref<index>
    %2 = memref.load %alloca[] : memref<index>
    %3 = index.add %2, %idx1
    memref.store %3, %alloca[] : memref<index>
    %4 = memref.load %alloca[] : memref<index>
    %5 = index.add %4, %idx1
    memref.store %5, %alloca[] : memref<index>
    %6 = memref.load %alloca[] : memref<index>
    %7 = index.add %6, %idx1
    memref.store %7, %alloca[] : memref<index>
    %8 = memref.load %alloca[] : memref<index>
    %9 = index.add %8, %idx1
    memref.store %9, %alloca[] : memref<index>
    %10 = memref.load %alloca[] : memref<index>
    %11 = index.add %10, %idx1
    memref.store %11, %alloca[] : memref<index>
    %12 = memref.load %alloca[] : memref<index>
    %13 = index.add %12, %idx1
    memref.store %13, %alloca[] : memref<index>
    %14 = memref.load %alloca[] : memref<index>
    %15 = index.add %14, %idx1
    memref.store %15, %alloca[] : memref<index>
    %16 = memref.load %alloca[] : memref<index>
    %17 = index.add %16, %idx1
    memref.store %17, %alloca[] : memref<index>
    %18 = memref.load %alloca[] : memref<index>
    %19 = index.add %18, %idx1
    memref.store %19, %alloca[] : memref<index>
    %20 = memref.load %alloca[] : memref<index>
    %21 = index.add %20, %idx1
    memref.store %21, %alloca[] : memref<index>
    %22 = memref.load %alloca[] : memref<index>
    %23 = index.add %22, %idx1
    memref.store %23, %alloca[] : memref<index>
    %24 = memref.load %alloca[] : memref<index>
    %25 = index.add %24, %idx1
    memref.store %25, %alloca[] : memref<index>
    %26 = memref.load %alloca[] : memref<index>
    %27 = index.add %26, %idx1
    memref.store %27, %alloca[] : memref<index>
    %28 = memref.load %alloca[] : memref<index>
    %29 = index.add %28, %idx1
    memref.store %29, %alloca[] : memref<index>
    %30 = memref.load %alloca[] : memref<index>
    %31 = index.add %30, %idx1
    memref.store %31, %alloca[] : memref<index>
    %32 = memref.load %alloca[] : memref<index>
    %33 = index.add %32, %idx1
    memref.store %33, %alloca[] : memref<index>
    %34 = memref.load %alloca[] : memref<index>
    %35 = index.add %34, %idx1
    memref.store %35, %alloca[] : memref<index>
    %36 = memref.load %alloca[] : memref<index>
    %37 = index.add %36, %idx1
    memref.store %37, %alloca[] : memref<index>
    %38 = memref.load %alloca[] : memref<index>
    %39 = index.add %38, %idx1
    memref.store %39, %alloca[] : memref<index>
    %40 = memref.load %alloca[] : memref<index>
    %41 = index.add %40, %idx1
    memref.store %41, %alloca[] : memref<index>
    %42 = memref.load %alloca[] : memref<index>
    %43 = index.add %42, %idx1
    memref.store %43, %alloca[] : memref<index>
    %44 = memref.load %alloca[] : memref<index>
    %45 = index.add %44, %idx1
    memref.store %45, %alloca[] : memref<index>
    %46 = memref.load %alloca[] : memref<index>
    %47 = index.add %46, %idx1
    memref.store %47, %alloca[] : memref<index>
    %48 = memref.load %alloca[] : memref<index>
    %49 = index.add %48, %idx1
    memref.store %49, %alloca[] : memref<index>
    %50 = memref.load %alloca[] : memref<index>
    %51 = index.add %50, %idx1
    memref.store %51, %alloca[] : memref<index>
    %52 = memref.load %alloca[] : memref<index>
    %53 = index.add %52, %idx1
    memref.store %53, %alloca[] : memref<index>
    %54 = memref.load %alloca[] : memref<index>
    %55 = index.add %54, %idx1
    memref.store %55, %alloca[] : memref<index>
    %56 = memref.load %alloca[] : memref<index>
    %57 = index.add %56, %idx1
    memref.store %57, %alloca[] : memref<index>
    %58 = memref.load %alloca[] : memref<index>
    %59 = index.add %58, %idx1
    memref.store %59, %alloca[] : memref<index>
    %60 = memref.load %alloca[] : memref<index>
    %61 = index.add %60, %idx1
    memref.store %61, %alloca[] : memref<index>
    %62 = memref.load %alloca[] : memref<index>
    %63 = index.add %62, %idx1
    memref.store %63, %alloca[] : memref<index>
    %64 = memref.load %alloca[] : memref<index>
    %65 = index.add %64, %idx1
    memref.store %65, %alloca[] : memref<index>
    %66 = memref.load %alloca[] : memref<index>
    %67 = index.add %66, %idx1
    memref.store %67, %alloca[] : memref<index>
    %68 = memref.load %alloca[] : memref<index>
    %69 = index.add %68, %idx1
    memref.store %69, %alloca[] : memref<index>
    %70 = memref.load %alloca[] : memref<index>
    %71 = index.add %70, %idx1
    memref.store %71, %alloca[] : memref<index>
    %72 = memref.load %alloca[] : memref<index>
    %73 = index.add %72, %idx1
    memref.store %73, %alloca[] : memref<index>
    %74 = memref.load %alloca[] : memref<index>
    %75 = index.add %74, %idx1
    memref.store %75, %alloca[] : memref<index>
    %76 = memref.load %alloca[] : memref<index>
    %77 = index.add %76, %idx1
    memref.store %77, %alloca[] : memref<index>
    %78 = memref.load %alloca[] : memref<index>
    %79 = index.add %78, %idx1
    memref.store %79, %alloca[] : memref<index>
    %80 = memref.load %alloca[] : memref<index>
    %81 = index.add %80, %idx1
    memref.store %81, %alloca[] : memref<index>
    %82 = memref.load %alloca[] : memref<index>
    %83 = index.add %82, %idx1
    memref.store %83, %alloca[] : memref<index>
    %84 = memref.load %alloca[] : memref<index>
    %85 = index.add %84, %idx1
    memref.store %85, %alloca[] : memref<index>
    %86 = memref.load %alloca[] : memref<index>
    %87 = index.add %86, %idx1
    memref.store %87, %alloca[] : memref<index>
    %88 = memref.load %alloca[] : memref<index>
    %89 = index.add %88, %idx1
    memref.store %89, %alloca[] : memref<index>
    %90 = memref.load %alloca[] : memref<index>
    %91 = index.add %90, %idx1
    memref.store %91, %alloca[] : memref<index>
    %92 = memref.load %alloca[] : memref<index>
    %93 = index.add %92, %idx1
    memref.store %93, %alloca[] : memref<index>
    %94 = memref.load %alloca[] : memref<index>
    %95 = index.add %94, %idx1
    memref.store %95, %alloca[] : memref<index>
    %96 = memref.load %alloca[] : memref<index>
    %97 = index.add %96, %idx1
    memref.store %97, %alloca[] : memref<index>
    %98 = memref.load %alloca[] : memref<index>
    %99 = index.add %98, %idx1
    memref.store %99, %alloca[] : memref<index>
    %100 = memref.load %alloca[] : memref<index>
    %101 = index.add %100, %idx1
    memref.store %101, %alloca[] : memref<index>
    %102 = memref.load %alloca[] : memref<index>
    %103 = index.add %102, %idx1
    memref.store %103, %alloca[] : memref<index>
    %104 = memref.load %alloca[] : memref<index>
    %105 = index.add %104, %idx1
    memref.store %105, %alloca[] : memref<index>
    %106 = memref.load %alloca[] : memref<index>
    %107 = index.add %106, %idx1
    memref.store %107, %alloca[] : memref<index>
    %108 = memref.load %alloca[] : memref<index>
    %109 = index.add %108, %idx1
    memref.store %109, %alloca[] : memref<index>
    %110 = memref.load %alloca[] : memref<index>
    %111 = index.add %110, %idx1
    memref.store %111, %alloca[] : memref<index>
    %112 = memref.load %alloca[] : memref<index>
    %113 = index.add %112, %idx1
    memref.store %113, %alloca[] : memref<index>
    %114 = memref.load %alloca[] : memref<index>
    %115 = index.add %114, %idx1
    memref.store %115, %alloca[] : memref<index>
    %116 = memref.load %alloca[] : memref<index>
    %117 = index.add %116, %idx1
    memref.store %117, %alloca[] : memref<index>
    %118 = memref.load %alloca[] : memref<index>
    %119 = index.add %118, %idx1
    memref.store %119, %alloca[] : memref<index>
    %120 = memref.load %alloca[] : memref<index>
    %121 = index.add %120, %idx1
    memref.store %121, %alloca[] : memref<index>
    %122 = memref.load %alloca[] : memref<index>
    %123 = index.add %122, %idx1
    memref.store %123, %alloca[] : memref<index>
    %124 = memref.load %alloca[] : memref<index>
    %125 = index.add %124, %idx1
    memref.store %125, %alloca[] : memref<index>
    %126 = memref.load %alloca[] : memref<index>
    %127 = index.add %126, %idx1
    memref.store %127, %alloca[] : memref<index>
    %128 = memref.load %alloca[] : memref<index>
    %129 = index.add %128, %idx1
    memref.store %129, %alloca[] : memref<index>
    %130 = memref.load %alloca[] : memref<index>
    %131 = index.add %130, %idx1
    memref.store %131, %alloca[] : memref<index>
    %132 = memref.load %alloca[] : memref<index>
    %133 = index.add %132, %idx1
    memref.store %133, %alloca[] : memref<index>
    %134 = memref.load %alloca[] : memref<index>
    %135 = index.add %134, %idx1
    memref.store %135, %alloca[] : memref<index>
    %136 = memref.load %alloca[] : memref<index>
    %137 = index.add %136, %idx1
    memref.store %137, %alloca[] : memref<index>
    %138 = memref.load %alloca[] : memref<index>
    %139 = index.add %138, %idx1
    memref.store %139, %alloca[] : memref<index>
    %140 = memref.load %alloca[] : memref<index>
    %141 = index.add %140, %idx1
    memref.store %141, %alloca[] : memref<index>
    %142 = memref.load %alloca[] : memref<index>
    %143 = index.add %142, %idx1
    memref.store %143, %alloca[] : memref<index>
    %144 = memref.load %alloca[] : memref<index>
    %145 = index.add %144, %idx1
    memref.store %145, %alloca[] : memref<index>
    %146 = memref.load %alloca[] : memref<index>
    %147 = index.add %146, %idx1
    memref.store %147, %alloca[] : memref<index>
    %148 = memref.load %alloca[] : memref<index>
    %149 = index.add %148, %idx1
    memref.store %149, %alloca[] : memref<index>
    %150 = memref.load %alloca[] : memref<index>
    %151 = index.add %150, %idx1
    memref.store %151, %alloca[] : memref<index>
    %152 = memref.load %alloca[] : memref<index>
    %153 = index.add %152, %idx1
    memref.store %153, %alloca[] : memref<index>
    %154 = memref.load %alloca[] : memref<index>
    %155 = index.add %154, %idx1
    memref.store %155, %alloca[] : memref<index>
    %156 = memref.load %alloca[] : memref<index>
    %157 = index.add %156, %idx1
    memref.store %157, %alloca[] : memref<index>
    %158 = memref.load %alloca[] : memref<index>
    %159 = index.add %158, %idx1
    memref.store %159, %alloca[] : memref<index>
    %160 = memref.load %alloca[] : memref<index>
    %161 = index.add %160, %idx1
    memref.store %161, %alloca[] : memref<index>
    %162 = memref.load %alloca[] : memref<index>
    %163 = index.add %162, %idx1
    memref.store %163, %alloca[] : memref<index>
    %164 = memref.load %alloca[] : memref<index>
    %165 = index.add %164, %idx1
    memref.store %165, %alloca[] : memref<index>
    %166 = memref.load %alloca[] : memref<index>
    %167 = index.add %166, %idx1
    memref.store %167, %alloca[] : memref<index>
    %168 = memref.load %alloca[] : memref<index>
    %169 = index.add %168, %idx1
    memref.store %169, %alloca[] : memref<index>
    %170 = memref.load %alloca[] : memref<index>
    %171 = index.add %170, %idx1
    memref.store %171, %alloca[] : memref<index>
    %172 = memref.load %alloca[] : memref<index>
    %173 = index.add %172, %idx1
    memref.store %173, %alloca[] : memref<index>
    %174 = memref.load %alloca[] : memref<index>
    %175 = index.add %174, %idx1
    memref.store %175, %alloca[] : memref<index>
    %176 = memref.load %alloca[] : memref<index>
    %177 = index.add %176, %idx1
    memref.store %177, %alloca[] : memref<index>
    %178 = memref.load %alloca[] : memref<index>
    %179 = index.add %178, %idx1
    memref.store %179, %alloca[] : memref<index>
    %180 = memref.load %alloca[] : memref<index>
    %181 = index.add %180, %idx1
    memref.store %181, %alloca[] : memref<index>
    %182 = memref.load %alloca[] : memref<index>
    %183 = index.add %182, %idx1
    memref.store %183, %alloca[] : memref<index>
    %184 = memref.load %alloca[] : memref<index>
    %185 = index.add %184, %idx1
    memref.store %185, %alloca[] : memref<index>
    %186 = memref.load %alloca[] : memref<index>
    %187 = index.add %186, %idx1
    memref.store %187, %alloca[] : memref<index>
    %188 = memref.load %alloca[] : memref<index>
    %189 = index.add %188, %idx1
    memref.store %189, %alloca[] : memref<index>
    %190 = memref.load %alloca[] : memref<index>
    %191 = index.add %190, %idx1
    memref.store %191, %alloca[] : memref<index>
    %192 = memref.load %alloca[] : memref<index>
    %193 = index.add %192, %idx1
    memref.store %193, %alloca[] : memref<index>
    %194 = memref.load %alloca[] : memref<index>
    %195 = index.add %194, %idx1
    memref.store %195, %alloca[] : memref<index>
    %196 = memref.load %alloca[] : memref<index>
    %197 = index.add %196, %idx1
    memref.store %197, %alloca[] : memref<index>
    %198 = memref.load %alloca[] : memref<index>
    %199 = index.add %198, %idx1
    memref.store %199, %alloca[] : memref<index>
    %200 = memref.load %alloca[] : memref<index>
    %201 = index.add %200, %idx1
    memref.store %201, %alloca[] : memref<index>
    %202 = memref.load %alloca[] : memref<index>
    %203 = index.add %202, %idx1
    memref.store %203, %alloca[] : memref<index>
    %204 = memref.load %alloca[] : memref<index>
    %205 = index.add %204, %idx1
    memref.store %205, %alloca[] : memref<index>
    %206 = memref.load %alloca[] : memref<index>
    %207 = index.add %206, %idx1
    memref.store %207, %alloca[] : memref<index>
    %208 = memref.load %alloca[] : memref<index>
    return %208 : index
  }
  func.func private @qnode_forward_0.quantum(%arg0: memref<4x8x3xf32>, %arg1: memref<8xf32>, %arg2: memref<?xf64>) -> memref<f64> attributes {gradient.qgrad = @qnode_forward_0.adjoint, passthrough = ["noinline"]} {
    %c0_i64 = arith.constant 0 : i64
    %idx1 = index.constant 1
    %idx0 = index.constant 0
    %alloca = memref.alloca() : memref<index>
    memref.store %idx0, %alloca[] : memref<index>
    quantum.device shots(%c0_i64) ["/home/pichau/QAgents/.venv/lib/python3.12/site-packages/pennylane_lightning/liblightning_gpu_catalyst.so", "LightningGPUSimulator", "{}"]
    %0 = quantum.alloc( 8) : !quantum.reg
    %1 = quantum.extract %0[ 7] : !quantum.reg -> !quantum.bit
    %2 = memref.load %alloca[] : memref<index>
    %3 = index.add %2, %idx1
    memref.store %3, %alloca[] : memref<index>
    %4 = memref.load %arg2[%2] : memref<?xf64>
    %out_qubits = quantum.custom "RY"(%4) %1 : !quantum.bit
    %5 = memref.load %alloca[] : memref<index>
    %6 = index.add %5, %idx1
    memref.store %6, %alloca[] : memref<index>
    %7 = memref.load %arg2[%5] : memref<?xf64>
    %out_qubits_0 = quantum.custom "RZ"(%7) %out_qubits : !quantum.bit
    %8 = memref.load %alloca[] : memref<index>
    %9 = index.add %8, %idx1
    memref.store %9, %alloca[] : memref<index>
    %10 = memref.load %arg2[%8] : memref<?xf64>
    %out_qubits_1 = quantum.custom "RY"(%10) %out_qubits_0 : !quantum.bit
    %11 = memref.load %alloca[] : memref<index>
    %12 = index.add %11, %idx1
    memref.store %12, %alloca[] : memref<index>
    %13 = memref.load %arg2[%11] : memref<?xf64>
    %out_qubits_2 = quantum.custom "RZ"(%13) %out_qubits_1 : !quantum.bit
    %14 = quantum.extract %0[ 6] : !quantum.reg -> !quantum.bit
    %15 = memref.load %alloca[] : memref<index>
    %16 = index.add %15, %idx1
    memref.store %16, %alloca[] : memref<index>
    %17 = memref.load %arg2[%15] : memref<?xf64>
    %out_qubits_3 = quantum.custom "RY"(%17) %14 : !quantum.bit
    %18 = memref.load %alloca[] : memref<index>
    %19 = index.add %18, %idx1
    memref.store %19, %alloca[] : memref<index>
    %20 = memref.load %arg2[%18] : memref<?xf64>
    %out_qubits_4 = quantum.custom "RZ"(%20) %out_qubits_3 : !quantum.bit
    %21 = memref.load %alloca[] : memref<index>
    %22 = index.add %21, %idx1
    memref.store %22, %alloca[] : memref<index>
    %23 = memref.load %arg2[%21] : memref<?xf64>
    %out_qubits_5 = quantum.custom "RY"(%23) %out_qubits_4 : !quantum.bit
    %24 = memref.load %alloca[] : memref<index>
    %25 = index.add %24, %idx1
    memref.store %25, %alloca[] : memref<index>
    %26 = memref.load %arg2[%24] : memref<?xf64>
    %out_qubits_6 = quantum.custom "RZ"(%26) %out_qubits_5 : !quantum.bit
    %27 = quantum.extract %0[ 5] : !quantum.reg -> !quantum.bit
    %28 = memref.load %alloca[] : memref<index>
    %29 = index.add %28, %idx1
    memref.store %29, %alloca[] : memref<index>
    %30 = memref.load %arg2[%28] : memref<?xf64>
    %out_qubits_7 = quantum.custom "RY"(%30) %27 : !quantum.bit
    %31 = memref.load %alloca[] : memref<index>
    %32 = index.add %31, %idx1
    memref.store %32, %alloca[] : memref<index>
    %33 = memref.load %arg2[%31] : memref<?xf64>
    %out_qubits_8 = quantum.custom "RZ"(%33) %out_qubits_7 : !quantum.bit
    %34 = memref.load %alloca[] : memref<index>
    %35 = index.add %34, %idx1
    memref.store %35, %alloca[] : memref<index>
    %36 = memref.load %arg2[%34] : memref<?xf64>
    %out_qubits_9 = quantum.custom "RY"(%36) %out_qubits_8 : !quantum.bit
    %37 = memref.load %alloca[] : memref<index>
    %38 = index.add %37, %idx1
    memref.store %38, %alloca[] : memref<index>
    %39 = memref.load %arg2[%37] : memref<?xf64>
    %out_qubits_10 = quantum.custom "RZ"(%39) %out_qubits_9 : !quantum.bit
    %40 = quantum.extract %0[ 4] : !quantum.reg -> !quantum.bit
    %41 = memref.load %alloca[] : memref<index>
    %42 = index.add %41, %idx1
    memref.store %42, %alloca[] : memref<index>
    %43 = memref.load %arg2[%41] : memref<?xf64>
    %out_qubits_11 = quantum.custom "RY"(%43) %40 : !quantum.bit
    %44 = memref.load %alloca[] : memref<index>
    %45 = index.add %44, %idx1
    memref.store %45, %alloca[] : memref<index>
    %46 = memref.load %arg2[%44] : memref<?xf64>
    %out_qubits_12 = quantum.custom "RZ"(%46) %out_qubits_11 : !quantum.bit
    %47 = memref.load %alloca[] : memref<index>
    %48 = index.add %47, %idx1
    memref.store %48, %alloca[] : memref<index>
    %49 = memref.load %arg2[%47] : memref<?xf64>
    %out_qubits_13 = quantum.custom "RY"(%49) %out_qubits_12 : !quantum.bit
    %50 = memref.load %alloca[] : memref<index>
    %51 = index.add %50, %idx1
    memref.store %51, %alloca[] : memref<index>
    %52 = memref.load %arg2[%50] : memref<?xf64>
    %out_qubits_14 = quantum.custom "RZ"(%52) %out_qubits_13 : !quantum.bit
    %53 = quantum.extract %0[ 3] : !quantum.reg -> !quantum.bit
    %54 = memref.load %alloca[] : memref<index>
    %55 = index.add %54, %idx1
    memref.store %55, %alloca[] : memref<index>
    %56 = memref.load %arg2[%54] : memref<?xf64>
    %out_qubits_15 = quantum.custom "RY"(%56) %53 : !quantum.bit
    %57 = memref.load %alloca[] : memref<index>
    %58 = index.add %57, %idx1
    memref.store %58, %alloca[] : memref<index>
    %59 = memref.load %arg2[%57] : memref<?xf64>
    %out_qubits_16 = quantum.custom "RZ"(%59) %out_qubits_15 : !quantum.bit
    %60 = memref.load %alloca[] : memref<index>
    %61 = index.add %60, %idx1
    memref.store %61, %alloca[] : memref<index>
    %62 = memref.load %arg2[%60] : memref<?xf64>
    %out_qubits_17 = quantum.custom "RY"(%62) %out_qubits_16 : !quantum.bit
    %63 = memref.load %alloca[] : memref<index>
    %64 = index.add %63, %idx1
    memref.store %64, %alloca[] : memref<index>
    %65 = memref.load %arg2[%63] : memref<?xf64>
    %out_qubits_18 = quantum.custom "RZ"(%65) %out_qubits_17 : !quantum.bit
    %66 = quantum.extract %0[ 2] : !quantum.reg -> !quantum.bit
    %67 = memref.load %alloca[] : memref<index>
    %68 = index.add %67, %idx1
    memref.store %68, %alloca[] : memref<index>
    %69 = memref.load %arg2[%67] : memref<?xf64>
    %out_qubits_19 = quantum.custom "RY"(%69) %66 : !quantum.bit
    %70 = memref.load %alloca[] : memref<index>
    %71 = index.add %70, %idx1
    memref.store %71, %alloca[] : memref<index>
    %72 = memref.load %arg2[%70] : memref<?xf64>
    %out_qubits_20 = quantum.custom "RZ"(%72) %out_qubits_19 : !quantum.bit
    %73 = memref.load %alloca[] : memref<index>
    %74 = index.add %73, %idx1
    memref.store %74, %alloca[] : memref<index>
    %75 = memref.load %arg2[%73] : memref<?xf64>
    %out_qubits_21 = quantum.custom "RY"(%75) %out_qubits_20 : !quantum.bit
    %76 = memref.load %alloca[] : memref<index>
    %77 = index.add %76, %idx1
    memref.store %77, %alloca[] : memref<index>
    %78 = memref.load %arg2[%76] : memref<?xf64>
    %out_qubits_22 = quantum.custom "RZ"(%78) %out_qubits_21 : !quantum.bit
    %79 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %80 = memref.load %alloca[] : memref<index>
    %81 = index.add %80, %idx1
    memref.store %81, %alloca[] : memref<index>
    %82 = memref.load %arg2[%80] : memref<?xf64>
    %out_qubits_23 = quantum.custom "RY"(%82) %79 : !quantum.bit
    %83 = memref.load %alloca[] : memref<index>
    %84 = index.add %83, %idx1
    memref.store %84, %alloca[] : memref<index>
    %85 = memref.load %arg2[%83] : memref<?xf64>
    %out_qubits_24 = quantum.custom "RZ"(%85) %out_qubits_23 : !quantum.bit
    %86 = memref.load %alloca[] : memref<index>
    %87 = index.add %86, %idx1
    memref.store %87, %alloca[] : memref<index>
    %88 = memref.load %arg2[%86] : memref<?xf64>
    %out_qubits_25 = quantum.custom "RY"(%88) %out_qubits_24 : !quantum.bit
    %89 = memref.load %alloca[] : memref<index>
    %90 = index.add %89, %idx1
    memref.store %90, %alloca[] : memref<index>
    %91 = memref.load %arg2[%89] : memref<?xf64>
    %out_qubits_26 = quantum.custom "RZ"(%91) %out_qubits_25 : !quantum.bit
    %92 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    %93 = memref.load %alloca[] : memref<index>
    %94 = index.add %93, %idx1
    memref.store %94, %alloca[] : memref<index>
    %95 = memref.load %arg2[%93] : memref<?xf64>
    %out_qubits_27 = quantum.custom "RY"(%95) %92 : !quantum.bit
    %96 = memref.load %alloca[] : memref<index>
    %97 = index.add %96, %idx1
    memref.store %97, %alloca[] : memref<index>
    %98 = memref.load %arg2[%96] : memref<?xf64>
    %out_qubits_28 = quantum.custom "RZ"(%98) %out_qubits_27 : !quantum.bit
    %99 = memref.load %alloca[] : memref<index>
    %100 = index.add %99, %idx1
    memref.store %100, %alloca[] : memref<index>
    %101 = memref.load %arg2[%99] : memref<?xf64>
    %out_qubits_29 = quantum.custom "RY"(%101) %out_qubits_28 : !quantum.bit
    %102 = memref.load %alloca[] : memref<index>
    %103 = index.add %102, %idx1
    memref.store %103, %alloca[] : memref<index>
    %104 = memref.load %arg2[%102] : memref<?xf64>
    %out_qubits_30 = quantum.custom "RZ"(%104) %out_qubits_29 : !quantum.bit
    %out_qubits_31:2 = quantum.custom "CNOT"() %out_qubits_26, %out_qubits_30 : !quantum.bit, !quantum.bit
    %out_qubits_32:2 = quantum.custom "CNOT"() %out_qubits_31#1, %out_qubits_22 : !quantum.bit, !quantum.bit
    %out_qubits_33:2 = quantum.custom "CNOT"() %out_qubits_32#1, %out_qubits_18 : !quantum.bit, !quantum.bit
    %out_qubits_34:2 = quantum.custom "CNOT"() %out_qubits_33#1, %out_qubits_14 : !quantum.bit, !quantum.bit
    %out_qubits_35:2 = quantum.custom "CNOT"() %out_qubits_34#1, %out_qubits_10 : !quantum.bit, !quantum.bit
    %out_qubits_36:2 = quantum.custom "CNOT"() %out_qubits_35#1, %out_qubits_6 : !quantum.bit, !quantum.bit
    %out_qubits_37:2 = quantum.custom "CNOT"() %out_qubits_36#1, %out_qubits_2 : !quantum.bit, !quantum.bit
    %105 = memref.load %alloca[] : memref<index>
    %106 = index.add %105, %idx1
    memref.store %106, %alloca[] : memref<index>
    %107 = memref.load %arg2[%105] : memref<?xf64>
    %out_qubits_38 = quantum.custom "RZ"(%107) %out_qubits_37#0 : !quantum.bit
    %108 = memref.load %alloca[] : memref<index>
    %109 = index.add %108, %idx1
    memref.store %109, %alloca[] : memref<index>
    %110 = memref.load %arg2[%108] : memref<?xf64>
    %out_qubits_39 = quantum.custom "RY"(%110) %out_qubits_38 : !quantum.bit
    %111 = memref.load %alloca[] : memref<index>
    %112 = index.add %111, %idx1
    memref.store %112, %alloca[] : memref<index>
    %113 = memref.load %arg2[%111] : memref<?xf64>
    %out_qubits_40 = quantum.custom "RZ"(%113) %out_qubits_39 : !quantum.bit
    %114 = memref.load %alloca[] : memref<index>
    %115 = index.add %114, %idx1
    memref.store %115, %alloca[] : memref<index>
    %116 = memref.load %arg2[%114] : memref<?xf64>
    %out_qubits_41 = quantum.custom "RZ"(%116) %out_qubits_35#0 : !quantum.bit
    %117 = memref.load %alloca[] : memref<index>
    %118 = index.add %117, %idx1
    memref.store %118, %alloca[] : memref<index>
    %119 = memref.load %arg2[%117] : memref<?xf64>
    %out_qubits_42 = quantum.custom "RY"(%119) %out_qubits_41 : !quantum.bit
    %120 = memref.load %alloca[] : memref<index>
    %121 = index.add %120, %idx1
    memref.store %121, %alloca[] : memref<index>
    %122 = memref.load %arg2[%120] : memref<?xf64>
    %out_qubits_43 = quantum.custom "RZ"(%122) %out_qubits_42 : !quantum.bit
    %out_qubits_44:2 = quantum.custom "CNOT"() %out_qubits_37#1, %out_qubits_31#0 : !quantum.bit, !quantum.bit
    %123 = memref.load %alloca[] : memref<index>
    %124 = index.add %123, %idx1
    memref.store %124, %alloca[] : memref<index>
    %125 = memref.load %arg2[%123] : memref<?xf64>
    %out_qubits_45 = quantum.custom "RZ"(%125) %out_qubits_44#1 : !quantum.bit
    %126 = memref.load %alloca[] : memref<index>
    %127 = index.add %126, %idx1
    memref.store %127, %alloca[] : memref<index>
    %128 = memref.load %arg2[%126] : memref<?xf64>
    %out_qubits_46 = quantum.custom "RY"(%128) %out_qubits_45 : !quantum.bit
    %129 = memref.load %alloca[] : memref<index>
    %130 = index.add %129, %idx1
    memref.store %130, %alloca[] : memref<index>
    %131 = memref.load %arg2[%129] : memref<?xf64>
    %out_qubits_47 = quantum.custom "RZ"(%131) %out_qubits_46 : !quantum.bit
    %132 = memref.load %alloca[] : memref<index>
    %133 = index.add %132, %idx1
    memref.store %133, %alloca[] : memref<index>
    %134 = memref.load %arg2[%132] : memref<?xf64>
    %out_qubits_48 = quantum.custom "RZ"(%134) %out_qubits_33#0 : !quantum.bit
    %135 = memref.load %alloca[] : memref<index>
    %136 = index.add %135, %idx1
    memref.store %136, %alloca[] : memref<index>
    %137 = memref.load %arg2[%135] : memref<?xf64>
    %out_qubits_49 = quantum.custom "RY"(%137) %out_qubits_48 : !quantum.bit
    %138 = memref.load %alloca[] : memref<index>
    %139 = index.add %138, %idx1
    memref.store %139, %alloca[] : memref<index>
    %140 = memref.load %arg2[%138] : memref<?xf64>
    %out_qubits_50 = quantum.custom "RZ"(%140) %out_qubits_49 : !quantum.bit
    %out_qubits_51:2 = quantum.custom "CNOT"() %out_qubits_47, %out_qubits_50 : !quantum.bit, !quantum.bit
    %out_qubits_52:2 = quantum.custom "CNOT"() %out_qubits_51#1, %out_qubits_43 : !quantum.bit, !quantum.bit
    %out_qubits_53:2 = quantum.custom "CNOT"() %out_qubits_52#1, %out_qubits_40 : !quantum.bit, !quantum.bit
    %out_qubits_54:2 = quantum.custom "CNOT"() %out_qubits_53#1, %out_qubits_51#0 : !quantum.bit, !quantum.bit
    %141 = memref.load %alloca[] : memref<index>
    %142 = index.add %141, %idx1
    memref.store %142, %alloca[] : memref<index>
    %143 = memref.load %arg2[%141] : memref<?xf64>
    %out_qubits_55 = quantum.custom "RZ"(%143) %out_qubits_54#1 : !quantum.bit
    %144 = memref.load %alloca[] : memref<index>
    %145 = index.add %144, %idx1
    memref.store %145, %alloca[] : memref<index>
    %146 = memref.load %arg2[%144] : memref<?xf64>
    %out_qubits_56 = quantum.custom "RY"(%146) %out_qubits_55 : !quantum.bit
    %147 = memref.load %alloca[] : memref<index>
    %148 = index.add %147, %idx1
    memref.store %148, %alloca[] : memref<index>
    %149 = memref.load %arg2[%147] : memref<?xf64>
    %out_qubits_57 = quantum.custom "RZ"(%149) %out_qubits_56 : !quantum.bit
    %150 = memref.load %alloca[] : memref<index>
    %151 = index.add %150, %idx1
    memref.store %151, %alloca[] : memref<index>
    %152 = memref.load %arg2[%150] : memref<?xf64>
    %out_qubits_58 = quantum.custom "RZ"(%152) %out_qubits_36#0 : !quantum.bit
    %153 = memref.load %alloca[] : memref<index>
    %154 = index.add %153, %idx1
    memref.store %154, %alloca[] : memref<index>
    %155 = memref.load %arg2[%153] : memref<?xf64>
    %out_qubits_59 = quantum.custom "RY"(%155) %out_qubits_58 : !quantum.bit
    %156 = memref.load %alloca[] : memref<index>
    %157 = index.add %156, %idx1
    memref.store %157, %alloca[] : memref<index>
    %158 = memref.load %arg2[%156] : memref<?xf64>
    %out_qubits_60 = quantum.custom "RZ"(%158) %out_qubits_59 : !quantum.bit
    %159 = memref.load %alloca[] : memref<index>
    %160 = index.add %159, %idx1
    memref.store %160, %alloca[] : memref<index>
    %161 = memref.load %arg2[%159] : memref<?xf64>
    %out_qubits_61 = quantum.custom "RZ"(%161) %out_qubits_32#0 : !quantum.bit
    %162 = memref.load %alloca[] : memref<index>
    %163 = index.add %162, %idx1
    memref.store %163, %alloca[] : memref<index>
    %164 = memref.load %arg2[%162] : memref<?xf64>
    %out_qubits_62 = quantum.custom "RY"(%164) %out_qubits_61 : !quantum.bit
    %165 = memref.load %alloca[] : memref<index>
    %166 = index.add %165, %idx1
    memref.store %166, %alloca[] : memref<index>
    %167 = memref.load %arg2[%165] : memref<?xf64>
    %out_qubits_63 = quantum.custom "RZ"(%167) %out_qubits_62 : !quantum.bit
    %168 = memref.load %alloca[] : memref<index>
    %169 = index.add %168, %idx1
    memref.store %169, %alloca[] : memref<index>
    %170 = memref.load %arg2[%168] : memref<?xf64>
    %out_qubits_64 = quantum.custom "RZ"(%170) %out_qubits_34#0 : !quantum.bit
    %171 = memref.load %alloca[] : memref<index>
    %172 = index.add %171, %idx1
    memref.store %172, %alloca[] : memref<index>
    %173 = memref.load %arg2[%171] : memref<?xf64>
    %out_qubits_65 = quantum.custom "RY"(%173) %out_qubits_64 : !quantum.bit
    %174 = memref.load %alloca[] : memref<index>
    %175 = index.add %174, %idx1
    memref.store %175, %alloca[] : memref<index>
    %176 = memref.load %arg2[%174] : memref<?xf64>
    %out_qubits_66 = quantum.custom "RZ"(%176) %out_qubits_65 : !quantum.bit
    %out_qubits_67:2 = quantum.custom "CNOT"() %out_qubits_63, %out_qubits_66 : !quantum.bit, !quantum.bit
    %out_qubits_68:2 = quantum.custom "CNOT"() %out_qubits_67#1, %out_qubits_60 : !quantum.bit, !quantum.bit
    %177 = memref.load %alloca[] : memref<index>
    %178 = index.add %177, %idx1
    memref.store %178, %alloca[] : memref<index>
    %179 = memref.load %arg2[%177] : memref<?xf64>
    %out_qubits_69 = quantum.custom "RZ"(%179) %out_qubits_68#0 : !quantum.bit
    %180 = memref.load %alloca[] : memref<index>
    %181 = index.add %180, %idx1
    memref.store %181, %alloca[] : memref<index>
    %182 = memref.load %arg2[%180] : memref<?xf64>
    %out_qubits_70 = quantum.custom "RY"(%182) %out_qubits_69 : !quantum.bit
    %183 = memref.load %alloca[] : memref<index>
    %184 = index.add %183, %idx1
    memref.store %184, %alloca[] : memref<index>
    %185 = memref.load %arg2[%183] : memref<?xf64>
    %out_qubits_71 = quantum.custom "RZ"(%185) %out_qubits_70 : !quantum.bit
    %out_qubits_72:2 = quantum.custom "CNOT"() %out_qubits_57, %out_qubits_71 : !quantum.bit, !quantum.bit
    %186 = memref.load %alloca[] : memref<index>
    %187 = index.add %186, %idx1
    memref.store %187, %alloca[] : memref<index>
    %188 = memref.load %arg2[%186] : memref<?xf64>
    %out_qubits_73 = quantum.custom "RZ"(%188) %out_qubits_52#0 : !quantum.bit
    %189 = memref.load %alloca[] : memref<index>
    %190 = index.add %189, %idx1
    memref.store %190, %alloca[] : memref<index>
    %191 = memref.load %arg2[%189] : memref<?xf64>
    %out_qubits_74 = quantum.custom "RY"(%191) %out_qubits_73 : !quantum.bit
    %192 = memref.load %alloca[] : memref<index>
    %193 = index.add %192, %idx1
    memref.store %193, %alloca[] : memref<index>
    %194 = memref.load %arg2[%192] : memref<?xf64>
    %out_qubits_75 = quantum.custom "RZ"(%194) %out_qubits_74 : !quantum.bit
    %195 = memref.load %alloca[] : memref<index>
    %196 = index.add %195, %idx1
    memref.store %196, %alloca[] : memref<index>
    %197 = memref.load %arg2[%195] : memref<?xf64>
    %out_qubits_76 = quantum.custom "RZ"(%197) %out_qubits_44#0 : !quantum.bit
    %198 = memref.load %alloca[] : memref<index>
    %199 = index.add %198, %idx1
    memref.store %199, %alloca[] : memref<index>
    %200 = memref.load %arg2[%198] : memref<?xf64>
    %out_qubits_77 = quantum.custom "RY"(%200) %out_qubits_76 : !quantum.bit
    %201 = memref.load %alloca[] : memref<index>
    %202 = index.add %201, %idx1
    memref.store %202, %alloca[] : memref<index>
    %203 = memref.load %arg2[%201] : memref<?xf64>
    %out_qubits_78 = quantum.custom "RZ"(%203) %out_qubits_77 : !quantum.bit
    %out_qubits_79:2 = quantum.custom "CNOT"() %out_qubits_68#1, %out_qubits_78 : !quantum.bit, !quantum.bit
    %204 = memref.load %alloca[] : memref<index>
    %205 = index.add %204, %idx1
    memref.store %205, %alloca[] : memref<index>
    %206 = memref.load %arg2[%204] : memref<?xf64>
    %out_qubits_80 = quantum.custom "RZ"(%206) %out_qubits_79#0 : !quantum.bit
    %207 = memref.load %alloca[] : memref<index>
    %208 = index.add %207, %idx1
    memref.store %208, %alloca[] : memref<index>
    %209 = memref.load %arg2[%207] : memref<?xf64>
    %out_qubits_81 = quantum.custom "RY"(%209) %out_qubits_80 : !quantum.bit
    %210 = memref.load %alloca[] : memref<index>
    %211 = index.add %210, %idx1
    memref.store %211, %alloca[] : memref<index>
    %212 = memref.load %arg2[%210] : memref<?xf64>
    %out_qubits_82 = quantum.custom "RZ"(%212) %out_qubits_81 : !quantum.bit
    %out_qubits_83:2 = quantum.custom "CNOT"() %out_qubits_75, %out_qubits_82 : !quantum.bit, !quantum.bit
    %out_qubits_84:2 = quantum.custom "CNOT"() %out_qubits_83#1, %out_qubits_72#0 : !quantum.bit, !quantum.bit
    %213 = memref.load %alloca[] : memref<index>
    %214 = index.add %213, %idx1
    memref.store %214, %alloca[] : memref<index>
    %215 = memref.load %arg2[%213] : memref<?xf64>
    %out_qubits_85 = quantum.custom "RZ"(%215) %out_qubits_84#1 : !quantum.bit
    %216 = memref.load %alloca[] : memref<index>
    %217 = index.add %216, %idx1
    memref.store %217, %alloca[] : memref<index>
    %218 = memref.load %arg2[%216] : memref<?xf64>
    %out_qubits_86 = quantum.custom "RY"(%218) %out_qubits_85 : !quantum.bit
    %219 = memref.load %alloca[] : memref<index>
    %220 = index.add %219, %idx1
    memref.store %220, %alloca[] : memref<index>
    %221 = memref.load %arg2[%219] : memref<?xf64>
    %out_qubits_87 = quantum.custom "RZ"(%221) %out_qubits_86 : !quantum.bit
    %out_qubits_88:2 = quantum.custom "CNOT"() %out_qubits_79#1, %out_qubits_67#0 : !quantum.bit, !quantum.bit
    %222 = memref.load %alloca[] : memref<index>
    %223 = index.add %222, %idx1
    memref.store %223, %alloca[] : memref<index>
    %224 = memref.load %arg2[%222] : memref<?xf64>
    %out_qubits_89 = quantum.custom "RZ"(%224) %out_qubits_88#0 : !quantum.bit
    %225 = memref.load %alloca[] : memref<index>
    %226 = index.add %225, %idx1
    memref.store %226, %alloca[] : memref<index>
    %227 = memref.load %arg2[%225] : memref<?xf64>
    %out_qubits_90 = quantum.custom "RY"(%227) %out_qubits_89 : !quantum.bit
    %228 = memref.load %alloca[] : memref<index>
    %229 = index.add %228, %idx1
    memref.store %229, %alloca[] : memref<index>
    %230 = memref.load %arg2[%228] : memref<?xf64>
    %out_qubits_91 = quantum.custom "RZ"(%230) %out_qubits_90 : !quantum.bit
    %231 = memref.load %alloca[] : memref<index>
    %232 = index.add %231, %idx1
    memref.store %232, %alloca[] : memref<index>
    %233 = memref.load %arg2[%231] : memref<?xf64>
    %out_qubits_92 = quantum.custom "RZ"(%233) %out_qubits_88#1 : !quantum.bit
    %234 = memref.load %alloca[] : memref<index>
    %235 = index.add %234, %idx1
    memref.store %235, %alloca[] : memref<index>
    %236 = memref.load %arg2[%234] : memref<?xf64>
    %out_qubits_93 = quantum.custom "RY"(%236) %out_qubits_92 : !quantum.bit
    %237 = memref.load %alloca[] : memref<index>
    %238 = index.add %237, %idx1
    memref.store %238, %alloca[] : memref<index>
    %239 = memref.load %arg2[%237] : memref<?xf64>
    %out_qubits_94 = quantum.custom "RZ"(%239) %out_qubits_93 : !quantum.bit
    %240 = memref.load %alloca[] : memref<index>
    %241 = index.add %240, %idx1
    memref.store %241, %alloca[] : memref<index>
    %242 = memref.load %arg2[%240] : memref<?xf64>
    %out_qubits_95 = quantum.custom "RZ"(%242) %out_qubits_53#0 : !quantum.bit
    %243 = memref.load %alloca[] : memref<index>
    %244 = index.add %243, %idx1
    memref.store %244, %alloca[] : memref<index>
    %245 = memref.load %arg2[%243] : memref<?xf64>
    %out_qubits_96 = quantum.custom "RY"(%245) %out_qubits_95 : !quantum.bit
    %246 = memref.load %alloca[] : memref<index>
    %247 = index.add %246, %idx1
    memref.store %247, %alloca[] : memref<index>
    %248 = memref.load %arg2[%246] : memref<?xf64>
    %out_qubits_97 = quantum.custom "RZ"(%248) %out_qubits_96 : !quantum.bit
    %out_qubits_98:2 = quantum.custom "CNOT"() %out_qubits_94, %out_qubits_97 : !quantum.bit, !quantum.bit
    %out_qubits_99:2 = quantum.custom "CNOT"() %out_qubits_98#1, %out_qubits_91 : !quantum.bit, !quantum.bit
    %249 = memref.load %alloca[] : memref<index>
    %250 = index.add %249, %idx1
    memref.store %250, %alloca[] : memref<index>
    %251 = memref.load %arg2[%249] : memref<?xf64>
    %out_qubits_100 = quantum.custom "RZ"(%251) %out_qubits_99#0 : !quantum.bit
    %252 = memref.load %alloca[] : memref<index>
    %253 = index.add %252, %idx1
    memref.store %253, %alloca[] : memref<index>
    %254 = memref.load %arg2[%252] : memref<?xf64>
    %out_qubits_101 = quantum.custom "RY"(%254) %out_qubits_100 : !quantum.bit
    %255 = memref.load %alloca[] : memref<index>
    %256 = index.add %255, %idx1
    memref.store %256, %alloca[] : memref<index>
    %257 = memref.load %arg2[%255] : memref<?xf64>
    %out_qubits_102 = quantum.custom "RZ"(%257) %out_qubits_101 : !quantum.bit
    %out_qubits_103:2 = quantum.custom "CNOT"() %out_qubits_87, %out_qubits_102 : !quantum.bit, !quantum.bit
    %out_qubits_104:2 = quantum.custom "CNOT"() %out_qubits_103#1, %out_qubits_103#0 : !quantum.bit, !quantum.bit
    %258 = memref.load %alloca[] : memref<index>
    %259 = index.add %258, %idx1
    memref.store %259, %alloca[] : memref<index>
    %260 = memref.load %arg2[%258] : memref<?xf64>
    %out_qubits_105 = quantum.custom "RZ"(%260) %out_qubits_54#0 : !quantum.bit
    %261 = memref.load %alloca[] : memref<index>
    %262 = index.add %261, %idx1
    memref.store %262, %alloca[] : memref<index>
    %263 = memref.load %arg2[%261] : memref<?xf64>
    %out_qubits_106 = quantum.custom "RY"(%263) %out_qubits_105 : !quantum.bit
    %264 = memref.load %alloca[] : memref<index>
    %265 = index.add %264, %idx1
    memref.store %265, %alloca[] : memref<index>
    %266 = memref.load %arg2[%264] : memref<?xf64>
    %out_qubits_107 = quantum.custom "RZ"(%266) %out_qubits_106 : !quantum.bit
    %out_qubits_108:2 = quantum.custom "CNOT"() %out_qubits_72#1, %out_qubits_107 : !quantum.bit, !quantum.bit
    %out_qubits_109:2 = quantum.custom "CNOT"() %out_qubits_108#1, %out_qubits_98#0 : !quantum.bit, !quantum.bit
    %267 = memref.load %alloca[] : memref<index>
    %268 = index.add %267, %idx1
    memref.store %268, %alloca[] : memref<index>
    %269 = memref.load %arg2[%267] : memref<?xf64>
    %out_qubits_110 = quantum.custom "RZ"(%269) %out_qubits_109#1 : !quantum.bit
    %270 = memref.load %alloca[] : memref<index>
    %271 = index.add %270, %idx1
    memref.store %271, %alloca[] : memref<index>
    %272 = memref.load %arg2[%270] : memref<?xf64>
    %out_qubits_111 = quantum.custom "RY"(%272) %out_qubits_110 : !quantum.bit
    %273 = memref.load %alloca[] : memref<index>
    %274 = index.add %273, %idx1
    memref.store %274, %alloca[] : memref<index>
    %275 = memref.load %arg2[%273] : memref<?xf64>
    %out_qubits_112 = quantum.custom "RZ"(%275) %out_qubits_111 : !quantum.bit
    %276 = memref.load %alloca[] : memref<index>
    %277 = index.add %276, %idx1
    memref.store %277, %alloca[] : memref<index>
    %278 = memref.load %arg2[%276] : memref<?xf64>
    %out_qubits_113 = quantum.custom "RZ"(%278) %out_qubits_84#0 : !quantum.bit
    %279 = memref.load %alloca[] : memref<index>
    %280 = index.add %279, %idx1
    memref.store %280, %alloca[] : memref<index>
    %281 = memref.load %arg2[%279] : memref<?xf64>
    %out_qubits_114 = quantum.custom "RY"(%281) %out_qubits_113 : !quantum.bit
    %282 = memref.load %alloca[] : memref<index>
    %283 = index.add %282, %idx1
    memref.store %283, %alloca[] : memref<index>
    %284 = memref.load %arg2[%282] : memref<?xf64>
    %out_qubits_115 = quantum.custom "RZ"(%284) %out_qubits_114 : !quantum.bit
    %out_qubits_116:2 = quantum.custom "CNOT"() %out_qubits_112, %out_qubits_115 : !quantum.bit, !quantum.bit
    %out_qubits_117:2 = quantum.custom "CNOT"() %out_qubits_116#1, %out_qubits_116#0 : !quantum.bit, !quantum.bit
    %out_qubits_118:2 = quantum.custom "CNOT"() %out_qubits_99#1, %out_qubits_83#0 : !quantum.bit, !quantum.bit
    %285 = memref.load %alloca[] : memref<index>
    %286 = index.add %285, %idx1
    memref.store %286, %alloca[] : memref<index>
    %287 = memref.load %arg2[%285] : memref<?xf64>
    %out_qubits_119 = quantum.custom "RZ"(%287) %out_qubits_118#1 : !quantum.bit
    %288 = memref.load %alloca[] : memref<index>
    %289 = index.add %288, %idx1
    memref.store %289, %alloca[] : memref<index>
    %290 = memref.load %arg2[%288] : memref<?xf64>
    %out_qubits_120 = quantum.custom "RY"(%290) %out_qubits_119 : !quantum.bit
    %291 = memref.load %alloca[] : memref<index>
    %292 = index.add %291, %idx1
    memref.store %292, %alloca[] : memref<index>
    %293 = memref.load %arg2[%291] : memref<?xf64>
    %out_qubits_121 = quantum.custom "RZ"(%293) %out_qubits_120 : !quantum.bit
    %294 = memref.load %alloca[] : memref<index>
    %295 = index.add %294, %idx1
    memref.store %295, %alloca[] : memref<index>
    %296 = memref.load %arg2[%294] : memref<?xf64>
    %out_qubits_122 = quantum.custom "RZ"(%296) %out_qubits_109#0 : !quantum.bit
    %297 = memref.load %alloca[] : memref<index>
    %298 = index.add %297, %idx1
    memref.store %298, %alloca[] : memref<index>
    %299 = memref.load %arg2[%297] : memref<?xf64>
    %out_qubits_123 = quantum.custom "RY"(%299) %out_qubits_122 : !quantum.bit
    %300 = memref.load %alloca[] : memref<index>
    %301 = index.add %300, %idx1
    memref.store %301, %alloca[] : memref<index>
    %302 = memref.load %arg2[%300] : memref<?xf64>
    %out_qubits_124 = quantum.custom "RZ"(%302) %out_qubits_123 : !quantum.bit
    %out_qubits_125:2 = quantum.custom "CNOT"() %out_qubits_121, %out_qubits_124 : !quantum.bit, !quantum.bit
    %out_qubits_126:2 = quantum.custom "CNOT"() %out_qubits_125#1, %out_qubits_125#0 : !quantum.bit, !quantum.bit
    %303 = memref.load %alloca[] : memref<index>
    %304 = index.add %303, %idx1
    memref.store %304, %alloca[] : memref<index>
    %305 = memref.load %arg2[%303] : memref<?xf64>
    %out_qubits_127 = quantum.custom "RZ"(%305) %out_qubits_108#0 : !quantum.bit
    %306 = memref.load %alloca[] : memref<index>
    %307 = index.add %306, %idx1
    memref.store %307, %alloca[] : memref<index>
    %308 = memref.load %arg2[%306] : memref<?xf64>
    %out_qubits_128 = quantum.custom "RY"(%308) %out_qubits_127 : !quantum.bit
    %309 = memref.load %alloca[] : memref<index>
    %310 = index.add %309, %idx1
    memref.store %310, %alloca[] : memref<index>
    %311 = memref.load %arg2[%309] : memref<?xf64>
    %out_qubits_129 = quantum.custom "RZ"(%311) %out_qubits_128 : !quantum.bit
    %312 = memref.load %alloca[] : memref<index>
    %313 = index.add %312, %idx1
    memref.store %313, %alloca[] : memref<index>
    %314 = memref.load %arg2[%312] : memref<?xf64>
    %out_qubits_130 = quantum.custom "RZ"(%314) %out_qubits_118#0 : !quantum.bit
    %315 = memref.load %alloca[] : memref<index>
    %316 = index.add %315, %idx1
    memref.store %316, %alloca[] : memref<index>
    %317 = memref.load %arg2[%315] : memref<?xf64>
    %out_qubits_131 = quantum.custom "RY"(%317) %out_qubits_130 : !quantum.bit
    %318 = memref.load %alloca[] : memref<index>
    %319 = index.add %318, %idx1
    memref.store %319, %alloca[] : memref<index>
    %320 = memref.load %arg2[%318] : memref<?xf64>
    %out_qubits_132 = quantum.custom "RZ"(%320) %out_qubits_131 : !quantum.bit
    %out_qubits_133:2 = quantum.custom "CNOT"() %out_qubits_129, %out_qubits_132 : !quantum.bit, !quantum.bit
    %out_qubits_134:2 = quantum.custom "CNOT"() %out_qubits_133#1, %out_qubits_133#0 : !quantum.bit, !quantum.bit
    %321 = quantum.namedobs %out_qubits_104#1[ PauliZ] : !quantum.obs
    %322 = quantum.expval %321 : f64
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<f64>
    memref.store %322, %alloc[] : memref<f64>
    %323 = quantum.insert %0[ 0], %out_qubits_104#1 : !quantum.reg, !quantum.bit
    %324 = quantum.insert %323[ 1], %out_qubits_117#1 : !quantum.reg, !quantum.bit
    %325 = quantum.insert %324[ 2], %out_qubits_126#1 : !quantum.reg, !quantum.bit
    %326 = quantum.insert %325[ 3], %out_qubits_134#1 : !quantum.reg, !quantum.bit
    %327 = quantum.insert %326[ 4], %out_qubits_104#0 : !quantum.reg, !quantum.bit
    %328 = quantum.insert %327[ 5], %out_qubits_117#0 : !quantum.reg, !quantum.bit
    %329 = quantum.insert %328[ 6], %out_qubits_126#0 : !quantum.reg, !quantum.bit
    %330 = quantum.insert %329[ 7], %out_qubits_134#0 : !quantum.reg, !quantum.bit
    quantum.dealloc %330 : !quantum.reg
    quantum.device_release
    return %alloc : memref<f64>
  }
  func.func private @qnode_forward_0.preprocess(%arg0: memref<4x8x3xf32>, %arg1: memref<8xf32>, %arg2: index) -> memref<f64> {
    %idx0 = index.constant 0
    %idx1 = index.constant 1
    %0 = memref.get_global @__constant_xf32 : memref<f32>
    %alloc = memref.alloc(%arg2) : memref<?xf64>
    %alloca = memref.alloca() : memref<index>
    memref.store %idx0, %alloca[] : memref<index>
    %subview = memref.subview %arg0[3, 0, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 74>>
    %collapse_shape = memref.collapse_shape %subview [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 74>> into memref<f32, strided<[], offset: 74>>
    %1 = memref.load %collapse_shape[] : memref<f32, strided<[], offset: 74>>
    %subview_0 = memref.subview %arg0[3, 0, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 73>>
    %collapse_shape_1 = memref.collapse_shape %subview_0 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 73>> into memref<f32, strided<[], offset: 73>>
    %2 = memref.load %collapse_shape_1[] : memref<f32, strided<[], offset: 73>>
    %subview_2 = memref.subview %arg0[3, 0, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 72>>
    %collapse_shape_3 = memref.collapse_shape %subview_2 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 72>> into memref<f32, strided<[], offset: 72>>
    %3 = memref.load %collapse_shape_3[] : memref<f32, strided<[], offset: 72>>
    %subview_4 = memref.subview %arg0[2, 0, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 50>>
    %collapse_shape_5 = memref.collapse_shape %subview_4 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 50>> into memref<f32, strided<[], offset: 50>>
    %4 = memref.load %collapse_shape_5[] : memref<f32, strided<[], offset: 50>>
    %subview_6 = memref.subview %arg0[2, 0, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 49>>
    %collapse_shape_7 = memref.collapse_shape %subview_6 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 49>> into memref<f32, strided<[], offset: 49>>
    %5 = memref.load %collapse_shape_7[] : memref<f32, strided<[], offset: 49>>
    %subview_8 = memref.subview %arg0[2, 0, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 48>>
    %collapse_shape_9 = memref.collapse_shape %subview_8 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 48>> into memref<f32, strided<[], offset: 48>>
    %6 = memref.load %collapse_shape_9[] : memref<f32, strided<[], offset: 48>>
    %subview_10 = memref.subview %arg0[1, 6, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 44>>
    %collapse_shape_11 = memref.collapse_shape %subview_10 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 44>> into memref<f32, strided<[], offset: 44>>
    %7 = memref.load %collapse_shape_11[] : memref<f32, strided<[], offset: 44>>
    %subview_12 = memref.subview %arg0[1, 6, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 43>>
    %collapse_shape_13 = memref.collapse_shape %subview_12 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 43>> into memref<f32, strided<[], offset: 43>>
    %8 = memref.load %collapse_shape_13[] : memref<f32, strided<[], offset: 43>>
    %subview_14 = memref.subview %arg0[1, 6, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 42>>
    %collapse_shape_15 = memref.collapse_shape %subview_14 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 42>> into memref<f32, strided<[], offset: 42>>
    %9 = memref.load %collapse_shape_15[] : memref<f32, strided<[], offset: 42>>
    %subview_16 = memref.subview %arg0[0, 7, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 23>>
    %collapse_shape_17 = memref.collapse_shape %subview_16 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 23>> into memref<f32, strided<[], offset: 23>>
    %10 = memref.load %collapse_shape_17[] : memref<f32, strided<[], offset: 23>>
    %subview_18 = memref.subview %arg0[0, 7, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 22>>
    %collapse_shape_19 = memref.collapse_shape %subview_18 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 22>> into memref<f32, strided<[], offset: 22>>
    %11 = memref.load %collapse_shape_19[] : memref<f32, strided<[], offset: 22>>
    %subview_20 = memref.subview %arg0[0, 7, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 21>>
    %collapse_shape_21 = memref.collapse_shape %subview_20 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 21>> into memref<f32, strided<[], offset: 21>>
    %12 = memref.load %collapse_shape_21[] : memref<f32, strided<[], offset: 21>>
    %alloc_22 = memref.alloc() {alignment = 64 : i64} : memref<8xf32>
    linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%0 : memref<f32>) outs(%alloc_22 : memref<8xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%alloc_22, %arg1 : memref<8xf32>, memref<8xf32>) outs(%alloc_22 : memref<8xf32>) {
    ^bb0(%in: f32, %in_207: f32, %out: f32):
      %418 = arith.mulf %in, %in_207 : f32
      linalg.yield %418 : f32
    }
    %subview_23 = memref.subview %alloc_22[7] [1] [1] : memref<8xf32> to memref<1xf32, strided<[1], offset: 7>>
    %collapse_shape_24 = memref.collapse_shape %subview_23 [] : memref<1xf32, strided<[1], offset: 7>> into memref<f32, strided<[], offset: 7>>
    %13 = memref.load %collapse_shape_24[] : memref<f32, strided<[], offset: 7>>
    %14 = arith.extf %13 : f32 to f64
    %15 = memref.load %alloca[] : memref<index>
    memref.store %14, %alloc[%15] : memref<?xf64>
    %16 = index.add %15, %idx1
    memref.store %16, %alloca[] : memref<index>
    %17 = arith.extf %12 : f32 to f64
    %18 = memref.load %alloca[] : memref<index>
    memref.store %17, %alloc[%18] : memref<?xf64>
    %19 = index.add %18, %idx1
    memref.store %19, %alloca[] : memref<index>
    %20 = arith.extf %11 : f32 to f64
    %21 = memref.load %alloca[] : memref<index>
    memref.store %20, %alloc[%21] : memref<?xf64>
    %22 = index.add %21, %idx1
    memref.store %22, %alloca[] : memref<index>
    %23 = arith.extf %10 : f32 to f64
    %24 = memref.load %alloca[] : memref<index>
    memref.store %23, %alloc[%24] : memref<?xf64>
    %25 = index.add %24, %idx1
    memref.store %25, %alloca[] : memref<index>
    %subview_25 = memref.subview %arg0[0, 6, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 20>>
    %collapse_shape_26 = memref.collapse_shape %subview_25 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 20>> into memref<f32, strided<[], offset: 20>>
    %26 = memref.load %collapse_shape_26[] : memref<f32, strided<[], offset: 20>>
    %subview_27 = memref.subview %arg0[0, 6, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 19>>
    %collapse_shape_28 = memref.collapse_shape %subview_27 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 19>> into memref<f32, strided<[], offset: 19>>
    %27 = memref.load %collapse_shape_28[] : memref<f32, strided<[], offset: 19>>
    %subview_29 = memref.subview %arg0[0, 6, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 18>>
    %collapse_shape_30 = memref.collapse_shape %subview_29 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 18>> into memref<f32, strided<[], offset: 18>>
    %28 = memref.load %collapse_shape_30[] : memref<f32, strided<[], offset: 18>>
    %subview_31 = memref.subview %alloc_22[6] [1] [1] : memref<8xf32> to memref<1xf32, strided<[1], offset: 6>>
    %collapse_shape_32 = memref.collapse_shape %subview_31 [] : memref<1xf32, strided<[1], offset: 6>> into memref<f32, strided<[], offset: 6>>
    %29 = memref.load %collapse_shape_32[] : memref<f32, strided<[], offset: 6>>
    %30 = arith.extf %29 : f32 to f64
    %31 = memref.load %alloca[] : memref<index>
    memref.store %30, %alloc[%31] : memref<?xf64>
    %32 = index.add %31, %idx1
    memref.store %32, %alloca[] : memref<index>
    %33 = arith.extf %28 : f32 to f64
    %34 = memref.load %alloca[] : memref<index>
    memref.store %33, %alloc[%34] : memref<?xf64>
    %35 = index.add %34, %idx1
    memref.store %35, %alloca[] : memref<index>
    %36 = arith.extf %27 : f32 to f64
    %37 = memref.load %alloca[] : memref<index>
    memref.store %36, %alloc[%37] : memref<?xf64>
    %38 = index.add %37, %idx1
    memref.store %38, %alloca[] : memref<index>
    %39 = arith.extf %26 : f32 to f64
    %40 = memref.load %alloca[] : memref<index>
    memref.store %39, %alloc[%40] : memref<?xf64>
    %41 = index.add %40, %idx1
    memref.store %41, %alloca[] : memref<index>
    %subview_33 = memref.subview %arg0[0, 5, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 17>>
    %collapse_shape_34 = memref.collapse_shape %subview_33 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 17>> into memref<f32, strided<[], offset: 17>>
    %42 = memref.load %collapse_shape_34[] : memref<f32, strided<[], offset: 17>>
    %subview_35 = memref.subview %arg0[0, 5, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 16>>
    %collapse_shape_36 = memref.collapse_shape %subview_35 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 16>> into memref<f32, strided<[], offset: 16>>
    %43 = memref.load %collapse_shape_36[] : memref<f32, strided<[], offset: 16>>
    %subview_37 = memref.subview %arg0[0, 5, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 15>>
    %collapse_shape_38 = memref.collapse_shape %subview_37 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 15>> into memref<f32, strided<[], offset: 15>>
    %44 = memref.load %collapse_shape_38[] : memref<f32, strided<[], offset: 15>>
    %subview_39 = memref.subview %alloc_22[5] [1] [1] : memref<8xf32> to memref<1xf32, strided<[1], offset: 5>>
    %collapse_shape_40 = memref.collapse_shape %subview_39 [] : memref<1xf32, strided<[1], offset: 5>> into memref<f32, strided<[], offset: 5>>
    %45 = memref.load %collapse_shape_40[] : memref<f32, strided<[], offset: 5>>
    %46 = arith.extf %45 : f32 to f64
    %47 = memref.load %alloca[] : memref<index>
    memref.store %46, %alloc[%47] : memref<?xf64>
    %48 = index.add %47, %idx1
    memref.store %48, %alloca[] : memref<index>
    %49 = arith.extf %44 : f32 to f64
    %50 = memref.load %alloca[] : memref<index>
    memref.store %49, %alloc[%50] : memref<?xf64>
    %51 = index.add %50, %idx1
    memref.store %51, %alloca[] : memref<index>
    %52 = arith.extf %43 : f32 to f64
    %53 = memref.load %alloca[] : memref<index>
    memref.store %52, %alloc[%53] : memref<?xf64>
    %54 = index.add %53, %idx1
    memref.store %54, %alloca[] : memref<index>
    %55 = arith.extf %42 : f32 to f64
    %56 = memref.load %alloca[] : memref<index>
    memref.store %55, %alloc[%56] : memref<?xf64>
    %57 = index.add %56, %idx1
    memref.store %57, %alloca[] : memref<index>
    %subview_41 = memref.subview %arg0[0, 4, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 14>>
    %collapse_shape_42 = memref.collapse_shape %subview_41 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 14>> into memref<f32, strided<[], offset: 14>>
    %58 = memref.load %collapse_shape_42[] : memref<f32, strided<[], offset: 14>>
    %subview_43 = memref.subview %arg0[0, 4, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 13>>
    %collapse_shape_44 = memref.collapse_shape %subview_43 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 13>> into memref<f32, strided<[], offset: 13>>
    %59 = memref.load %collapse_shape_44[] : memref<f32, strided<[], offset: 13>>
    %subview_45 = memref.subview %arg0[0, 4, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 12>>
    %collapse_shape_46 = memref.collapse_shape %subview_45 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 12>> into memref<f32, strided<[], offset: 12>>
    %60 = memref.load %collapse_shape_46[] : memref<f32, strided<[], offset: 12>>
    %subview_47 = memref.subview %alloc_22[4] [1] [1] : memref<8xf32> to memref<1xf32, strided<[1], offset: 4>>
    %collapse_shape_48 = memref.collapse_shape %subview_47 [] : memref<1xf32, strided<[1], offset: 4>> into memref<f32, strided<[], offset: 4>>
    %61 = memref.load %collapse_shape_48[] : memref<f32, strided<[], offset: 4>>
    %62 = arith.extf %61 : f32 to f64
    %63 = memref.load %alloca[] : memref<index>
    memref.store %62, %alloc[%63] : memref<?xf64>
    %64 = index.add %63, %idx1
    memref.store %64, %alloca[] : memref<index>
    %65 = arith.extf %60 : f32 to f64
    %66 = memref.load %alloca[] : memref<index>
    memref.store %65, %alloc[%66] : memref<?xf64>
    %67 = index.add %66, %idx1
    memref.store %67, %alloca[] : memref<index>
    %68 = arith.extf %59 : f32 to f64
    %69 = memref.load %alloca[] : memref<index>
    memref.store %68, %alloc[%69] : memref<?xf64>
    %70 = index.add %69, %idx1
    memref.store %70, %alloca[] : memref<index>
    %71 = arith.extf %58 : f32 to f64
    %72 = memref.load %alloca[] : memref<index>
    memref.store %71, %alloc[%72] : memref<?xf64>
    %73 = index.add %72, %idx1
    memref.store %73, %alloca[] : memref<index>
    %subview_49 = memref.subview %arg0[0, 3, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 11>>
    %collapse_shape_50 = memref.collapse_shape %subview_49 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 11>> into memref<f32, strided<[], offset: 11>>
    %74 = memref.load %collapse_shape_50[] : memref<f32, strided<[], offset: 11>>
    %subview_51 = memref.subview %arg0[0, 3, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 10>>
    %collapse_shape_52 = memref.collapse_shape %subview_51 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 10>> into memref<f32, strided<[], offset: 10>>
    %75 = memref.load %collapse_shape_52[] : memref<f32, strided<[], offset: 10>>
    %subview_53 = memref.subview %arg0[0, 3, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 9>>
    %collapse_shape_54 = memref.collapse_shape %subview_53 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 9>> into memref<f32, strided<[], offset: 9>>
    %76 = memref.load %collapse_shape_54[] : memref<f32, strided<[], offset: 9>>
    %subview_55 = memref.subview %alloc_22[3] [1] [1] : memref<8xf32> to memref<1xf32, strided<[1], offset: 3>>
    %collapse_shape_56 = memref.collapse_shape %subview_55 [] : memref<1xf32, strided<[1], offset: 3>> into memref<f32, strided<[], offset: 3>>
    %77 = memref.load %collapse_shape_56[] : memref<f32, strided<[], offset: 3>>
    %78 = arith.extf %77 : f32 to f64
    %79 = memref.load %alloca[] : memref<index>
    memref.store %78, %alloc[%79] : memref<?xf64>
    %80 = index.add %79, %idx1
    memref.store %80, %alloca[] : memref<index>
    %81 = arith.extf %76 : f32 to f64
    %82 = memref.load %alloca[] : memref<index>
    memref.store %81, %alloc[%82] : memref<?xf64>
    %83 = index.add %82, %idx1
    memref.store %83, %alloca[] : memref<index>
    %84 = arith.extf %75 : f32 to f64
    %85 = memref.load %alloca[] : memref<index>
    memref.store %84, %alloc[%85] : memref<?xf64>
    %86 = index.add %85, %idx1
    memref.store %86, %alloca[] : memref<index>
    %87 = arith.extf %74 : f32 to f64
    %88 = memref.load %alloca[] : memref<index>
    memref.store %87, %alloc[%88] : memref<?xf64>
    %89 = index.add %88, %idx1
    memref.store %89, %alloca[] : memref<index>
    %subview_57 = memref.subview %arg0[0, 2, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 8>>
    %collapse_shape_58 = memref.collapse_shape %subview_57 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 8>> into memref<f32, strided<[], offset: 8>>
    %90 = memref.load %collapse_shape_58[] : memref<f32, strided<[], offset: 8>>
    %subview_59 = memref.subview %arg0[0, 2, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 7>>
    %collapse_shape_60 = memref.collapse_shape %subview_59 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 7>> into memref<f32, strided<[], offset: 7>>
    %91 = memref.load %collapse_shape_60[] : memref<f32, strided<[], offset: 7>>
    %subview_61 = memref.subview %arg0[0, 2, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 6>>
    %collapse_shape_62 = memref.collapse_shape %subview_61 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 6>> into memref<f32, strided<[], offset: 6>>
    %92 = memref.load %collapse_shape_62[] : memref<f32, strided<[], offset: 6>>
    %subview_63 = memref.subview %alloc_22[2] [1] [1] : memref<8xf32> to memref<1xf32, strided<[1], offset: 2>>
    %collapse_shape_64 = memref.collapse_shape %subview_63 [] : memref<1xf32, strided<[1], offset: 2>> into memref<f32, strided<[], offset: 2>>
    %93 = memref.load %collapse_shape_64[] : memref<f32, strided<[], offset: 2>>
    %94 = arith.extf %93 : f32 to f64
    %95 = memref.load %alloca[] : memref<index>
    memref.store %94, %alloc[%95] : memref<?xf64>
    %96 = index.add %95, %idx1
    memref.store %96, %alloca[] : memref<index>
    %97 = arith.extf %92 : f32 to f64
    %98 = memref.load %alloca[] : memref<index>
    memref.store %97, %alloc[%98] : memref<?xf64>
    %99 = index.add %98, %idx1
    memref.store %99, %alloca[] : memref<index>
    %100 = arith.extf %91 : f32 to f64
    %101 = memref.load %alloca[] : memref<index>
    memref.store %100, %alloc[%101] : memref<?xf64>
    %102 = index.add %101, %idx1
    memref.store %102, %alloca[] : memref<index>
    %103 = arith.extf %90 : f32 to f64
    %104 = memref.load %alloca[] : memref<index>
    memref.store %103, %alloc[%104] : memref<?xf64>
    %105 = index.add %104, %idx1
    memref.store %105, %alloca[] : memref<index>
    %subview_65 = memref.subview %arg0[0, 0, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 2>>
    %collapse_shape_66 = memref.collapse_shape %subview_65 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 2>> into memref<f32, strided<[], offset: 2>>
    %106 = memref.load %collapse_shape_66[] : memref<f32, strided<[], offset: 2>>
    %subview_67 = memref.subview %arg0[0, 0, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 1>>
    %collapse_shape_68 = memref.collapse_shape %subview_67 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 1>> into memref<f32, strided<[], offset: 1>>
    %107 = memref.load %collapse_shape_68[] : memref<f32, strided<[], offset: 1>>
    %subview_69 = memref.subview %arg0[0, 0, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1]>>
    %collapse_shape_70 = memref.collapse_shape %subview_69 [] : memref<1x1x1xf32, strided<[24, 3, 1]>> into memref<f32, strided<[]>>
    %108 = memref.load %collapse_shape_70[] : memref<f32, strided<[]>>
    %subview_71 = memref.subview %alloc_22[0] [1] [1] : memref<8xf32> to memref<1xf32, strided<[1]>>
    %collapse_shape_72 = memref.collapse_shape %subview_71 [] : memref<1xf32, strided<[1]>> into memref<f32>
    %109 = memref.load %collapse_shape_72[] : memref<f32>
    %110 = arith.extf %109 : f32 to f64
    %111 = memref.load %alloca[] : memref<index>
    memref.store %110, %alloc[%111] : memref<?xf64>
    %112 = index.add %111, %idx1
    memref.store %112, %alloca[] : memref<index>
    %113 = arith.extf %108 : f32 to f64
    %114 = memref.load %alloca[] : memref<index>
    memref.store %113, %alloc[%114] : memref<?xf64>
    %115 = index.add %114, %idx1
    memref.store %115, %alloca[] : memref<index>
    %116 = arith.extf %107 : f32 to f64
    %117 = memref.load %alloca[] : memref<index>
    memref.store %116, %alloc[%117] : memref<?xf64>
    %118 = index.add %117, %idx1
    memref.store %118, %alloca[] : memref<index>
    %119 = arith.extf %106 : f32 to f64
    %120 = memref.load %alloca[] : memref<index>
    memref.store %119, %alloc[%120] : memref<?xf64>
    %121 = index.add %120, %idx1
    memref.store %121, %alloca[] : memref<index>
    %subview_73 = memref.subview %arg0[0, 1, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 5>>
    %collapse_shape_74 = memref.collapse_shape %subview_73 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 5>> into memref<f32, strided<[], offset: 5>>
    %122 = memref.load %collapse_shape_74[] : memref<f32, strided<[], offset: 5>>
    %subview_75 = memref.subview %arg0[0, 1, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 4>>
    %collapse_shape_76 = memref.collapse_shape %subview_75 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 4>> into memref<f32, strided<[], offset: 4>>
    %123 = memref.load %collapse_shape_76[] : memref<f32, strided<[], offset: 4>>
    %subview_77 = memref.subview %arg0[0, 1, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 3>>
    %collapse_shape_78 = memref.collapse_shape %subview_77 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 3>> into memref<f32, strided<[], offset: 3>>
    %124 = memref.load %collapse_shape_78[] : memref<f32, strided<[], offset: 3>>
    %subview_79 = memref.subview %alloc_22[1] [1] [1] : memref<8xf32> to memref<1xf32, strided<[1], offset: 1>>
    %collapse_shape_80 = memref.collapse_shape %subview_79 [] : memref<1xf32, strided<[1], offset: 1>> into memref<f32, strided<[], offset: 1>>
    %125 = memref.load %collapse_shape_80[] : memref<f32, strided<[], offset: 1>>
    memref.dealloc %alloc_22 : memref<8xf32>
    %126 = arith.extf %125 : f32 to f64
    %127 = memref.load %alloca[] : memref<index>
    memref.store %126, %alloc[%127] : memref<?xf64>
    %128 = index.add %127, %idx1
    memref.store %128, %alloca[] : memref<index>
    %129 = arith.extf %124 : f32 to f64
    %130 = memref.load %alloca[] : memref<index>
    memref.store %129, %alloc[%130] : memref<?xf64>
    %131 = index.add %130, %idx1
    memref.store %131, %alloca[] : memref<index>
    %132 = arith.extf %123 : f32 to f64
    %133 = memref.load %alloca[] : memref<index>
    memref.store %132, %alloc[%133] : memref<?xf64>
    %134 = index.add %133, %idx1
    memref.store %134, %alloca[] : memref<index>
    %135 = arith.extf %122 : f32 to f64
    %136 = memref.load %alloca[] : memref<index>
    memref.store %135, %alloc[%136] : memref<?xf64>
    %137 = index.add %136, %idx1
    memref.store %137, %alloca[] : memref<index>
    %138 = arith.extf %9 : f32 to f64
    %139 = memref.load %alloca[] : memref<index>
    memref.store %138, %alloc[%139] : memref<?xf64>
    %140 = index.add %139, %idx1
    memref.store %140, %alloca[] : memref<index>
    %141 = arith.extf %8 : f32 to f64
    %142 = memref.load %alloca[] : memref<index>
    memref.store %141, %alloc[%142] : memref<?xf64>
    %143 = index.add %142, %idx1
    memref.store %143, %alloca[] : memref<index>
    %144 = arith.extf %7 : f32 to f64
    %145 = memref.load %alloca[] : memref<index>
    memref.store %144, %alloc[%145] : memref<?xf64>
    %146 = index.add %145, %idx1
    memref.store %146, %alloca[] : memref<index>
    %subview_81 = memref.subview %arg0[1, 4, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 38>>
    %collapse_shape_82 = memref.collapse_shape %subview_81 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 38>> into memref<f32, strided<[], offset: 38>>
    %147 = memref.load %collapse_shape_82[] : memref<f32, strided<[], offset: 38>>
    %subview_83 = memref.subview %arg0[1, 4, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 37>>
    %collapse_shape_84 = memref.collapse_shape %subview_83 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 37>> into memref<f32, strided<[], offset: 37>>
    %148 = memref.load %collapse_shape_84[] : memref<f32, strided<[], offset: 37>>
    %subview_85 = memref.subview %arg0[1, 4, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 36>>
    %collapse_shape_86 = memref.collapse_shape %subview_85 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 36>> into memref<f32, strided<[], offset: 36>>
    %149 = memref.load %collapse_shape_86[] : memref<f32, strided<[], offset: 36>>
    %150 = arith.extf %149 : f32 to f64
    %151 = memref.load %alloca[] : memref<index>
    memref.store %150, %alloc[%151] : memref<?xf64>
    %152 = index.add %151, %idx1
    memref.store %152, %alloca[] : memref<index>
    %153 = arith.extf %148 : f32 to f64
    %154 = memref.load %alloca[] : memref<index>
    memref.store %153, %alloc[%154] : memref<?xf64>
    %155 = index.add %154, %idx1
    memref.store %155, %alloca[] : memref<index>
    %156 = arith.extf %147 : f32 to f64
    %157 = memref.load %alloca[] : memref<index>
    memref.store %156, %alloc[%157] : memref<?xf64>
    %158 = index.add %157, %idx1
    memref.store %158, %alloca[] : memref<index>
    %subview_87 = memref.subview %arg0[1, 0, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 26>>
    %collapse_shape_88 = memref.collapse_shape %subview_87 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 26>> into memref<f32, strided<[], offset: 26>>
    %159 = memref.load %collapse_shape_88[] : memref<f32, strided<[], offset: 26>>
    %subview_89 = memref.subview %arg0[1, 0, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 25>>
    %collapse_shape_90 = memref.collapse_shape %subview_89 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 25>> into memref<f32, strided<[], offset: 25>>
    %160 = memref.load %collapse_shape_90[] : memref<f32, strided<[], offset: 25>>
    %subview_91 = memref.subview %arg0[1, 0, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 24>>
    %collapse_shape_92 = memref.collapse_shape %subview_91 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 24>> into memref<f32, strided<[], offset: 24>>
    %161 = memref.load %collapse_shape_92[] : memref<f32, strided<[], offset: 24>>
    %162 = arith.extf %161 : f32 to f64
    %163 = memref.load %alloca[] : memref<index>
    memref.store %162, %alloc[%163] : memref<?xf64>
    %164 = index.add %163, %idx1
    memref.store %164, %alloca[] : memref<index>
    %165 = arith.extf %160 : f32 to f64
    %166 = memref.load %alloca[] : memref<index>
    memref.store %165, %alloc[%166] : memref<?xf64>
    %167 = index.add %166, %idx1
    memref.store %167, %alloca[] : memref<index>
    %168 = arith.extf %159 : f32 to f64
    %169 = memref.load %alloca[] : memref<index>
    memref.store %168, %alloc[%169] : memref<?xf64>
    %170 = index.add %169, %idx1
    memref.store %170, %alloca[] : memref<index>
    %subview_93 = memref.subview %arg0[1, 2, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 32>>
    %collapse_shape_94 = memref.collapse_shape %subview_93 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 32>> into memref<f32, strided<[], offset: 32>>
    %171 = memref.load %collapse_shape_94[] : memref<f32, strided<[], offset: 32>>
    %subview_95 = memref.subview %arg0[1, 2, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 31>>
    %collapse_shape_96 = memref.collapse_shape %subview_95 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 31>> into memref<f32, strided<[], offset: 31>>
    %172 = memref.load %collapse_shape_96[] : memref<f32, strided<[], offset: 31>>
    %subview_97 = memref.subview %arg0[1, 2, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 30>>
    %collapse_shape_98 = memref.collapse_shape %subview_97 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 30>> into memref<f32, strided<[], offset: 30>>
    %173 = memref.load %collapse_shape_98[] : memref<f32, strided<[], offset: 30>>
    %174 = arith.extf %173 : f32 to f64
    %175 = memref.load %alloca[] : memref<index>
    memref.store %174, %alloc[%175] : memref<?xf64>
    %176 = index.add %175, %idx1
    memref.store %176, %alloca[] : memref<index>
    %177 = arith.extf %172 : f32 to f64
    %178 = memref.load %alloca[] : memref<index>
    memref.store %177, %alloc[%178] : memref<?xf64>
    %179 = index.add %178, %idx1
    memref.store %179, %alloca[] : memref<index>
    %180 = arith.extf %171 : f32 to f64
    %181 = memref.load %alloca[] : memref<index>
    memref.store %180, %alloc[%181] : memref<?xf64>
    %182 = index.add %181, %idx1
    memref.store %182, %alloca[] : memref<index>
    %183 = arith.extf %6 : f32 to f64
    %184 = memref.load %alloca[] : memref<index>
    memref.store %183, %alloc[%184] : memref<?xf64>
    %185 = index.add %184, %idx1
    memref.store %185, %alloca[] : memref<index>
    %186 = arith.extf %5 : f32 to f64
    %187 = memref.load %alloca[] : memref<index>
    memref.store %186, %alloc[%187] : memref<?xf64>
    %188 = index.add %187, %idx1
    memref.store %188, %alloca[] : memref<index>
    %189 = arith.extf %4 : f32 to f64
    %190 = memref.load %alloca[] : memref<index>
    memref.store %189, %alloc[%190] : memref<?xf64>
    %191 = index.add %190, %idx1
    memref.store %191, %alloca[] : memref<index>
    %subview_99 = memref.subview %arg0[2, 3, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 59>>
    %collapse_shape_100 = memref.collapse_shape %subview_99 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 59>> into memref<f32, strided<[], offset: 59>>
    %192 = memref.load %collapse_shape_100[] : memref<f32, strided<[], offset: 59>>
    %subview_101 = memref.subview %arg0[2, 3, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 58>>
    %collapse_shape_102 = memref.collapse_shape %subview_101 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 58>> into memref<f32, strided<[], offset: 58>>
    %193 = memref.load %collapse_shape_102[] : memref<f32, strided<[], offset: 58>>
    %subview_103 = memref.subview %arg0[2, 3, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 57>>
    %collapse_shape_104 = memref.collapse_shape %subview_103 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 57>> into memref<f32, strided<[], offset: 57>>
    %194 = memref.load %collapse_shape_104[] : memref<f32, strided<[], offset: 57>>
    %subview_105 = memref.subview %arg0[1, 5, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 41>>
    %collapse_shape_106 = memref.collapse_shape %subview_105 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 41>> into memref<f32, strided<[], offset: 41>>
    %195 = memref.load %collapse_shape_106[] : memref<f32, strided<[], offset: 41>>
    %subview_107 = memref.subview %arg0[1, 5, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 40>>
    %collapse_shape_108 = memref.collapse_shape %subview_107 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 40>> into memref<f32, strided<[], offset: 40>>
    %196 = memref.load %collapse_shape_108[] : memref<f32, strided<[], offset: 40>>
    %subview_109 = memref.subview %arg0[1, 5, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 39>>
    %collapse_shape_110 = memref.collapse_shape %subview_109 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 39>> into memref<f32, strided<[], offset: 39>>
    %197 = memref.load %collapse_shape_110[] : memref<f32, strided<[], offset: 39>>
    %198 = arith.extf %197 : f32 to f64
    %199 = memref.load %alloca[] : memref<index>
    memref.store %198, %alloc[%199] : memref<?xf64>
    %200 = index.add %199, %idx1
    memref.store %200, %alloca[] : memref<index>
    %201 = arith.extf %196 : f32 to f64
    %202 = memref.load %alloca[] : memref<index>
    memref.store %201, %alloc[%202] : memref<?xf64>
    %203 = index.add %202, %idx1
    memref.store %203, %alloca[] : memref<index>
    %204 = arith.extf %195 : f32 to f64
    %205 = memref.load %alloca[] : memref<index>
    memref.store %204, %alloc[%205] : memref<?xf64>
    %206 = index.add %205, %idx1
    memref.store %206, %alloca[] : memref<index>
    %subview_111 = memref.subview %arg0[1, 1, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 29>>
    %collapse_shape_112 = memref.collapse_shape %subview_111 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 29>> into memref<f32, strided<[], offset: 29>>
    %207 = memref.load %collapse_shape_112[] : memref<f32, strided<[], offset: 29>>
    %subview_113 = memref.subview %arg0[1, 1, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 28>>
    %collapse_shape_114 = memref.collapse_shape %subview_113 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 28>> into memref<f32, strided<[], offset: 28>>
    %208 = memref.load %collapse_shape_114[] : memref<f32, strided<[], offset: 28>>
    %subview_115 = memref.subview %arg0[1, 1, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 27>>
    %collapse_shape_116 = memref.collapse_shape %subview_115 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 27>> into memref<f32, strided<[], offset: 27>>
    %209 = memref.load %collapse_shape_116[] : memref<f32, strided<[], offset: 27>>
    %210 = arith.extf %209 : f32 to f64
    %211 = memref.load %alloca[] : memref<index>
    memref.store %210, %alloc[%211] : memref<?xf64>
    %212 = index.add %211, %idx1
    memref.store %212, %alloca[] : memref<index>
    %213 = arith.extf %208 : f32 to f64
    %214 = memref.load %alloca[] : memref<index>
    memref.store %213, %alloc[%214] : memref<?xf64>
    %215 = index.add %214, %idx1
    memref.store %215, %alloca[] : memref<index>
    %216 = arith.extf %207 : f32 to f64
    %217 = memref.load %alloca[] : memref<index>
    memref.store %216, %alloc[%217] : memref<?xf64>
    %218 = index.add %217, %idx1
    memref.store %218, %alloca[] : memref<index>
    %subview_117 = memref.subview %arg0[1, 3, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 35>>
    %collapse_shape_118 = memref.collapse_shape %subview_117 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 35>> into memref<f32, strided<[], offset: 35>>
    %219 = memref.load %collapse_shape_118[] : memref<f32, strided<[], offset: 35>>
    %subview_119 = memref.subview %arg0[1, 3, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 34>>
    %collapse_shape_120 = memref.collapse_shape %subview_119 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 34>> into memref<f32, strided<[], offset: 34>>
    %220 = memref.load %collapse_shape_120[] : memref<f32, strided<[], offset: 34>>
    %subview_121 = memref.subview %arg0[1, 3, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 33>>
    %collapse_shape_122 = memref.collapse_shape %subview_121 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 33>> into memref<f32, strided<[], offset: 33>>
    %221 = memref.load %collapse_shape_122[] : memref<f32, strided<[], offset: 33>>
    %222 = arith.extf %221 : f32 to f64
    %223 = memref.load %alloca[] : memref<index>
    memref.store %222, %alloc[%223] : memref<?xf64>
    %224 = index.add %223, %idx1
    memref.store %224, %alloca[] : memref<index>
    %225 = arith.extf %220 : f32 to f64
    %226 = memref.load %alloca[] : memref<index>
    memref.store %225, %alloc[%226] : memref<?xf64>
    %227 = index.add %226, %idx1
    memref.store %227, %alloca[] : memref<index>
    %228 = arith.extf %219 : f32 to f64
    %229 = memref.load %alloca[] : memref<index>
    memref.store %228, %alloc[%229] : memref<?xf64>
    %230 = index.add %229, %idx1
    memref.store %230, %alloca[] : memref<index>
    %231 = arith.extf %194 : f32 to f64
    %232 = memref.load %alloca[] : memref<index>
    memref.store %231, %alloc[%232] : memref<?xf64>
    %233 = index.add %232, %idx1
    memref.store %233, %alloca[] : memref<index>
    %234 = arith.extf %193 : f32 to f64
    %235 = memref.load %alloca[] : memref<index>
    memref.store %234, %alloc[%235] : memref<?xf64>
    %236 = index.add %235, %idx1
    memref.store %236, %alloca[] : memref<index>
    %237 = arith.extf %192 : f32 to f64
    %238 = memref.load %alloca[] : memref<index>
    memref.store %237, %alloc[%238] : memref<?xf64>
    %239 = index.add %238, %idx1
    memref.store %239, %alloca[] : memref<index>
    %subview_123 = memref.subview %arg0[2, 2, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 56>>
    %collapse_shape_124 = memref.collapse_shape %subview_123 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 56>> into memref<f32, strided<[], offset: 56>>
    %240 = memref.load %collapse_shape_124[] : memref<f32, strided<[], offset: 56>>
    %subview_125 = memref.subview %arg0[2, 2, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 55>>
    %collapse_shape_126 = memref.collapse_shape %subview_125 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 55>> into memref<f32, strided<[], offset: 55>>
    %241 = memref.load %collapse_shape_126[] : memref<f32, strided<[], offset: 55>>
    %subview_127 = memref.subview %arg0[2, 2, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 54>>
    %collapse_shape_128 = memref.collapse_shape %subview_127 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 54>> into memref<f32, strided<[], offset: 54>>
    %242 = memref.load %collapse_shape_128[] : memref<f32, strided<[], offset: 54>>
    %243 = arith.extf %242 : f32 to f64
    %244 = memref.load %alloca[] : memref<index>
    memref.store %243, %alloc[%244] : memref<?xf64>
    %245 = index.add %244, %idx1
    memref.store %245, %alloca[] : memref<index>
    %246 = arith.extf %241 : f32 to f64
    %247 = memref.load %alloca[] : memref<index>
    memref.store %246, %alloc[%247] : memref<?xf64>
    %248 = index.add %247, %idx1
    memref.store %248, %alloca[] : memref<index>
    %249 = arith.extf %240 : f32 to f64
    %250 = memref.load %alloca[] : memref<index>
    memref.store %249, %alloc[%250] : memref<?xf64>
    %251 = index.add %250, %idx1
    memref.store %251, %alloca[] : memref<index>
    %subview_129 = memref.subview %arg0[2, 5, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 65>>
    %collapse_shape_130 = memref.collapse_shape %subview_129 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 65>> into memref<f32, strided<[], offset: 65>>
    %252 = memref.load %collapse_shape_130[] : memref<f32, strided<[], offset: 65>>
    %subview_131 = memref.subview %arg0[2, 5, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 64>>
    %collapse_shape_132 = memref.collapse_shape %subview_131 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 64>> into memref<f32, strided<[], offset: 64>>
    %253 = memref.load %collapse_shape_132[] : memref<f32, strided<[], offset: 64>>
    %subview_133 = memref.subview %arg0[2, 5, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 63>>
    %collapse_shape_134 = memref.collapse_shape %subview_133 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 63>> into memref<f32, strided<[], offset: 63>>
    %254 = memref.load %collapse_shape_134[] : memref<f32, strided<[], offset: 63>>
    %subview_135 = memref.subview %arg0[1, 7, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 47>>
    %collapse_shape_136 = memref.collapse_shape %subview_135 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 47>> into memref<f32, strided<[], offset: 47>>
    %255 = memref.load %collapse_shape_136[] : memref<f32, strided<[], offset: 47>>
    %subview_137 = memref.subview %arg0[1, 7, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 46>>
    %collapse_shape_138 = memref.collapse_shape %subview_137 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 46>> into memref<f32, strided<[], offset: 46>>
    %256 = memref.load %collapse_shape_138[] : memref<f32, strided<[], offset: 46>>
    %subview_139 = memref.subview %arg0[1, 7, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 45>>
    %collapse_shape_140 = memref.collapse_shape %subview_139 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 45>> into memref<f32, strided<[], offset: 45>>
    %257 = memref.load %collapse_shape_140[] : memref<f32, strided<[], offset: 45>>
    %258 = arith.extf %257 : f32 to f64
    %259 = memref.load %alloca[] : memref<index>
    memref.store %258, %alloc[%259] : memref<?xf64>
    %260 = index.add %259, %idx1
    memref.store %260, %alloca[] : memref<index>
    %261 = arith.extf %256 : f32 to f64
    %262 = memref.load %alloca[] : memref<index>
    memref.store %261, %alloc[%262] : memref<?xf64>
    %263 = index.add %262, %idx1
    memref.store %263, %alloca[] : memref<index>
    %264 = arith.extf %255 : f32 to f64
    %265 = memref.load %alloca[] : memref<index>
    memref.store %264, %alloc[%265] : memref<?xf64>
    %266 = index.add %265, %idx1
    memref.store %266, %alloca[] : memref<index>
    %267 = arith.extf %254 : f32 to f64
    %268 = memref.load %alloca[] : memref<index>
    memref.store %267, %alloc[%268] : memref<?xf64>
    %269 = index.add %268, %idx1
    memref.store %269, %alloca[] : memref<index>
    %270 = arith.extf %253 : f32 to f64
    %271 = memref.load %alloca[] : memref<index>
    memref.store %270, %alloc[%271] : memref<?xf64>
    %272 = index.add %271, %idx1
    memref.store %272, %alloca[] : memref<index>
    %273 = arith.extf %252 : f32 to f64
    %274 = memref.load %alloca[] : memref<index>
    memref.store %273, %alloc[%274] : memref<?xf64>
    %275 = index.add %274, %idx1
    memref.store %275, %alloca[] : memref<index>
    %276 = arith.extf %3 : f32 to f64
    %277 = memref.load %alloca[] : memref<index>
    memref.store %276, %alloc[%277] : memref<?xf64>
    %278 = index.add %277, %idx1
    memref.store %278, %alloca[] : memref<index>
    %279 = arith.extf %2 : f32 to f64
    %280 = memref.load %alloca[] : memref<index>
    memref.store %279, %alloc[%280] : memref<?xf64>
    %281 = index.add %280, %idx1
    memref.store %281, %alloca[] : memref<index>
    %282 = arith.extf %1 : f32 to f64
    %283 = memref.load %alloca[] : memref<index>
    memref.store %282, %alloc[%283] : memref<?xf64>
    %284 = index.add %283, %idx1
    memref.store %284, %alloca[] : memref<index>
    %subview_141 = memref.subview %arg0[3, 4, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 86>>
    %collapse_shape_142 = memref.collapse_shape %subview_141 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 86>> into memref<f32, strided<[], offset: 86>>
    %285 = memref.load %collapse_shape_142[] : memref<f32, strided<[], offset: 86>>
    %subview_143 = memref.subview %arg0[3, 4, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 85>>
    %collapse_shape_144 = memref.collapse_shape %subview_143 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 85>> into memref<f32, strided<[], offset: 85>>
    %286 = memref.load %collapse_shape_144[] : memref<f32, strided<[], offset: 85>>
    %subview_145 = memref.subview %arg0[3, 4, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 84>>
    %collapse_shape_146 = memref.collapse_shape %subview_145 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 84>> into memref<f32, strided<[], offset: 84>>
    %287 = memref.load %collapse_shape_146[] : memref<f32, strided<[], offset: 84>>
    %subview_147 = memref.subview %arg0[2, 7, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 71>>
    %collapse_shape_148 = memref.collapse_shape %subview_147 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 71>> into memref<f32, strided<[], offset: 71>>
    %288 = memref.load %collapse_shape_148[] : memref<f32, strided<[], offset: 71>>
    %subview_149 = memref.subview %arg0[2, 7, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 70>>
    %collapse_shape_150 = memref.collapse_shape %subview_149 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 70>> into memref<f32, strided<[], offset: 70>>
    %289 = memref.load %collapse_shape_150[] : memref<f32, strided<[], offset: 70>>
    %subview_151 = memref.subview %arg0[2, 7, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 69>>
    %collapse_shape_152 = memref.collapse_shape %subview_151 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 69>> into memref<f32, strided<[], offset: 69>>
    %290 = memref.load %collapse_shape_152[] : memref<f32, strided<[], offset: 69>>
    %291 = arith.extf %290 : f32 to f64
    %292 = memref.load %alloca[] : memref<index>
    memref.store %291, %alloc[%292] : memref<?xf64>
    %293 = index.add %292, %idx1
    memref.store %293, %alloca[] : memref<index>
    %294 = arith.extf %289 : f32 to f64
    %295 = memref.load %alloca[] : memref<index>
    memref.store %294, %alloc[%295] : memref<?xf64>
    %296 = index.add %295, %idx1
    memref.store %296, %alloca[] : memref<index>
    %297 = arith.extf %288 : f32 to f64
    %298 = memref.load %alloca[] : memref<index>
    memref.store %297, %alloc[%298] : memref<?xf64>
    %299 = index.add %298, %idx1
    memref.store %299, %alloca[] : memref<index>
    %subview_153 = memref.subview %arg0[2, 1, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 53>>
    %collapse_shape_154 = memref.collapse_shape %subview_153 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 53>> into memref<f32, strided<[], offset: 53>>
    %300 = memref.load %collapse_shape_154[] : memref<f32, strided<[], offset: 53>>
    %subview_155 = memref.subview %arg0[2, 1, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 52>>
    %collapse_shape_156 = memref.collapse_shape %subview_155 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 52>> into memref<f32, strided<[], offset: 52>>
    %301 = memref.load %collapse_shape_156[] : memref<f32, strided<[], offset: 52>>
    %subview_157 = memref.subview %arg0[2, 1, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 51>>
    %collapse_shape_158 = memref.collapse_shape %subview_157 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 51>> into memref<f32, strided<[], offset: 51>>
    %302 = memref.load %collapse_shape_158[] : memref<f32, strided<[], offset: 51>>
    %303 = arith.extf %302 : f32 to f64
    %304 = memref.load %alloca[] : memref<index>
    memref.store %303, %alloc[%304] : memref<?xf64>
    %305 = index.add %304, %idx1
    memref.store %305, %alloca[] : memref<index>
    %306 = arith.extf %301 : f32 to f64
    %307 = memref.load %alloca[] : memref<index>
    memref.store %306, %alloc[%307] : memref<?xf64>
    %308 = index.add %307, %idx1
    memref.store %308, %alloca[] : memref<index>
    %309 = arith.extf %300 : f32 to f64
    %310 = memref.load %alloca[] : memref<index>
    memref.store %309, %alloc[%310] : memref<?xf64>
    %311 = index.add %310, %idx1
    memref.store %311, %alloca[] : memref<index>
    %subview_159 = memref.subview %arg0[2, 4, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 62>>
    %collapse_shape_160 = memref.collapse_shape %subview_159 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 62>> into memref<f32, strided<[], offset: 62>>
    %312 = memref.load %collapse_shape_160[] : memref<f32, strided<[], offset: 62>>
    %subview_161 = memref.subview %arg0[2, 4, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 61>>
    %collapse_shape_162 = memref.collapse_shape %subview_161 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 61>> into memref<f32, strided<[], offset: 61>>
    %313 = memref.load %collapse_shape_162[] : memref<f32, strided<[], offset: 61>>
    %subview_163 = memref.subview %arg0[2, 4, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 60>>
    %collapse_shape_164 = memref.collapse_shape %subview_163 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 60>> into memref<f32, strided<[], offset: 60>>
    %314 = memref.load %collapse_shape_164[] : memref<f32, strided<[], offset: 60>>
    %315 = arith.extf %314 : f32 to f64
    %316 = memref.load %alloca[] : memref<index>
    memref.store %315, %alloc[%316] : memref<?xf64>
    %317 = index.add %316, %idx1
    memref.store %317, %alloca[] : memref<index>
    %318 = arith.extf %313 : f32 to f64
    %319 = memref.load %alloca[] : memref<index>
    memref.store %318, %alloc[%319] : memref<?xf64>
    %320 = index.add %319, %idx1
    memref.store %320, %alloca[] : memref<index>
    %321 = arith.extf %312 : f32 to f64
    %322 = memref.load %alloca[] : memref<index>
    memref.store %321, %alloc[%322] : memref<?xf64>
    %323 = index.add %322, %idx1
    memref.store %323, %alloca[] : memref<index>
    %324 = arith.extf %287 : f32 to f64
    %325 = memref.load %alloca[] : memref<index>
    memref.store %324, %alloc[%325] : memref<?xf64>
    %326 = index.add %325, %idx1
    memref.store %326, %alloca[] : memref<index>
    %327 = arith.extf %286 : f32 to f64
    %328 = memref.load %alloca[] : memref<index>
    memref.store %327, %alloc[%328] : memref<?xf64>
    %329 = index.add %328, %idx1
    memref.store %329, %alloca[] : memref<index>
    %330 = arith.extf %285 : f32 to f64
    %331 = memref.load %alloca[] : memref<index>
    memref.store %330, %alloc[%331] : memref<?xf64>
    %332 = index.add %331, %idx1
    memref.store %332, %alloca[] : memref<index>
    %subview_165 = memref.subview %arg0[3, 1, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 77>>
    %collapse_shape_166 = memref.collapse_shape %subview_165 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 77>> into memref<f32, strided<[], offset: 77>>
    %333 = memref.load %collapse_shape_166[] : memref<f32, strided<[], offset: 77>>
    %subview_167 = memref.subview %arg0[3, 1, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 76>>
    %collapse_shape_168 = memref.collapse_shape %subview_167 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 76>> into memref<f32, strided<[], offset: 76>>
    %334 = memref.load %collapse_shape_168[] : memref<f32, strided<[], offset: 76>>
    %subview_169 = memref.subview %arg0[3, 1, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 75>>
    %collapse_shape_170 = memref.collapse_shape %subview_169 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 75>> into memref<f32, strided<[], offset: 75>>
    %335 = memref.load %collapse_shape_170[] : memref<f32, strided<[], offset: 75>>
    %subview_171 = memref.subview %arg0[2, 6, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 68>>
    %collapse_shape_172 = memref.collapse_shape %subview_171 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 68>> into memref<f32, strided<[], offset: 68>>
    %336 = memref.load %collapse_shape_172[] : memref<f32, strided<[], offset: 68>>
    %subview_173 = memref.subview %arg0[2, 6, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 67>>
    %collapse_shape_174 = memref.collapse_shape %subview_173 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 67>> into memref<f32, strided<[], offset: 67>>
    %337 = memref.load %collapse_shape_174[] : memref<f32, strided<[], offset: 67>>
    %subview_175 = memref.subview %arg0[2, 6, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 66>>
    %collapse_shape_176 = memref.collapse_shape %subview_175 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 66>> into memref<f32, strided<[], offset: 66>>
    %338 = memref.load %collapse_shape_176[] : memref<f32, strided<[], offset: 66>>
    %339 = arith.extf %338 : f32 to f64
    %340 = memref.load %alloca[] : memref<index>
    memref.store %339, %alloc[%340] : memref<?xf64>
    %341 = index.add %340, %idx1
    memref.store %341, %alloca[] : memref<index>
    %342 = arith.extf %337 : f32 to f64
    %343 = memref.load %alloca[] : memref<index>
    memref.store %342, %alloc[%343] : memref<?xf64>
    %344 = index.add %343, %idx1
    memref.store %344, %alloca[] : memref<index>
    %345 = arith.extf %336 : f32 to f64
    %346 = memref.load %alloca[] : memref<index>
    memref.store %345, %alloc[%346] : memref<?xf64>
    %347 = index.add %346, %idx1
    memref.store %347, %alloca[] : memref<index>
    %348 = arith.extf %335 : f32 to f64
    %349 = memref.load %alloca[] : memref<index>
    memref.store %348, %alloc[%349] : memref<?xf64>
    %350 = index.add %349, %idx1
    memref.store %350, %alloca[] : memref<index>
    %351 = arith.extf %334 : f32 to f64
    %352 = memref.load %alloca[] : memref<index>
    memref.store %351, %alloc[%352] : memref<?xf64>
    %353 = index.add %352, %idx1
    memref.store %353, %alloca[] : memref<index>
    %354 = arith.extf %333 : f32 to f64
    %355 = memref.load %alloca[] : memref<index>
    memref.store %354, %alloc[%355] : memref<?xf64>
    %356 = index.add %355, %idx1
    memref.store %356, %alloca[] : memref<index>
    %subview_177 = memref.subview %arg0[3, 5, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 89>>
    %collapse_shape_178 = memref.collapse_shape %subview_177 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 89>> into memref<f32, strided<[], offset: 89>>
    %357 = memref.load %collapse_shape_178[] : memref<f32, strided<[], offset: 89>>
    %subview_179 = memref.subview %arg0[3, 5, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 88>>
    %collapse_shape_180 = memref.collapse_shape %subview_179 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 88>> into memref<f32, strided<[], offset: 88>>
    %358 = memref.load %collapse_shape_180[] : memref<f32, strided<[], offset: 88>>
    %subview_181 = memref.subview %arg0[3, 5, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 87>>
    %collapse_shape_182 = memref.collapse_shape %subview_181 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 87>> into memref<f32, strided<[], offset: 87>>
    %359 = memref.load %collapse_shape_182[] : memref<f32, strided<[], offset: 87>>
    %360 = arith.extf %359 : f32 to f64
    %361 = memref.load %alloca[] : memref<index>
    memref.store %360, %alloc[%361] : memref<?xf64>
    %362 = index.add %361, %idx1
    memref.store %362, %alloca[] : memref<index>
    %363 = arith.extf %358 : f32 to f64
    %364 = memref.load %alloca[] : memref<index>
    memref.store %363, %alloc[%364] : memref<?xf64>
    %365 = index.add %364, %idx1
    memref.store %365, %alloca[] : memref<index>
    %366 = arith.extf %357 : f32 to f64
    %367 = memref.load %alloca[] : memref<index>
    memref.store %366, %alloc[%367] : memref<?xf64>
    %368 = index.add %367, %idx1
    memref.store %368, %alloca[] : memref<index>
    %subview_183 = memref.subview %arg0[3, 2, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 80>>
    %collapse_shape_184 = memref.collapse_shape %subview_183 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 80>> into memref<f32, strided<[], offset: 80>>
    %369 = memref.load %collapse_shape_184[] : memref<f32, strided<[], offset: 80>>
    %subview_185 = memref.subview %arg0[3, 2, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 79>>
    %collapse_shape_186 = memref.collapse_shape %subview_185 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 79>> into memref<f32, strided<[], offset: 79>>
    %370 = memref.load %collapse_shape_186[] : memref<f32, strided<[], offset: 79>>
    %subview_187 = memref.subview %arg0[3, 2, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 78>>
    %collapse_shape_188 = memref.collapse_shape %subview_187 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 78>> into memref<f32, strided<[], offset: 78>>
    %371 = memref.load %collapse_shape_188[] : memref<f32, strided<[], offset: 78>>
    %372 = arith.extf %371 : f32 to f64
    %373 = memref.load %alloca[] : memref<index>
    memref.store %372, %alloc[%373] : memref<?xf64>
    %374 = index.add %373, %idx1
    memref.store %374, %alloca[] : memref<index>
    %375 = arith.extf %370 : f32 to f64
    %376 = memref.load %alloca[] : memref<index>
    memref.store %375, %alloc[%376] : memref<?xf64>
    %377 = index.add %376, %idx1
    memref.store %377, %alloca[] : memref<index>
    %378 = arith.extf %369 : f32 to f64
    %379 = memref.load %alloca[] : memref<index>
    memref.store %378, %alloc[%379] : memref<?xf64>
    %380 = index.add %379, %idx1
    memref.store %380, %alloca[] : memref<index>
    %subview_189 = memref.subview %arg0[3, 6, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 92>>
    %collapse_shape_190 = memref.collapse_shape %subview_189 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 92>> into memref<f32, strided<[], offset: 92>>
    %381 = memref.load %collapse_shape_190[] : memref<f32, strided<[], offset: 92>>
    %subview_191 = memref.subview %arg0[3, 6, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 91>>
    %collapse_shape_192 = memref.collapse_shape %subview_191 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 91>> into memref<f32, strided<[], offset: 91>>
    %382 = memref.load %collapse_shape_192[] : memref<f32, strided<[], offset: 91>>
    %subview_193 = memref.subview %arg0[3, 6, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 90>>
    %collapse_shape_194 = memref.collapse_shape %subview_193 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 90>> into memref<f32, strided<[], offset: 90>>
    %383 = memref.load %collapse_shape_194[] : memref<f32, strided<[], offset: 90>>
    %384 = arith.extf %383 : f32 to f64
    %385 = memref.load %alloca[] : memref<index>
    memref.store %384, %alloc[%385] : memref<?xf64>
    %386 = index.add %385, %idx1
    memref.store %386, %alloca[] : memref<index>
    %387 = arith.extf %382 : f32 to f64
    %388 = memref.load %alloca[] : memref<index>
    memref.store %387, %alloc[%388] : memref<?xf64>
    %389 = index.add %388, %idx1
    memref.store %389, %alloca[] : memref<index>
    %390 = arith.extf %381 : f32 to f64
    %391 = memref.load %alloca[] : memref<index>
    memref.store %390, %alloc[%391] : memref<?xf64>
    %392 = index.add %391, %idx1
    memref.store %392, %alloca[] : memref<index>
    %subview_195 = memref.subview %arg0[3, 3, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 83>>
    %collapse_shape_196 = memref.collapse_shape %subview_195 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 83>> into memref<f32, strided<[], offset: 83>>
    %393 = memref.load %collapse_shape_196[] : memref<f32, strided<[], offset: 83>>
    %subview_197 = memref.subview %arg0[3, 3, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 82>>
    %collapse_shape_198 = memref.collapse_shape %subview_197 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 82>> into memref<f32, strided<[], offset: 82>>
    %394 = memref.load %collapse_shape_198[] : memref<f32, strided<[], offset: 82>>
    %subview_199 = memref.subview %arg0[3, 3, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 81>>
    %collapse_shape_200 = memref.collapse_shape %subview_199 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 81>> into memref<f32, strided<[], offset: 81>>
    %395 = memref.load %collapse_shape_200[] : memref<f32, strided<[], offset: 81>>
    %396 = arith.extf %395 : f32 to f64
    %397 = memref.load %alloca[] : memref<index>
    memref.store %396, %alloc[%397] : memref<?xf64>
    %398 = index.add %397, %idx1
    memref.store %398, %alloca[] : memref<index>
    %399 = arith.extf %394 : f32 to f64
    %400 = memref.load %alloca[] : memref<index>
    memref.store %399, %alloc[%400] : memref<?xf64>
    %401 = index.add %400, %idx1
    memref.store %401, %alloca[] : memref<index>
    %402 = arith.extf %393 : f32 to f64
    %403 = memref.load %alloca[] : memref<index>
    memref.store %402, %alloc[%403] : memref<?xf64>
    %404 = index.add %403, %idx1
    memref.store %404, %alloca[] : memref<index>
    %subview_201 = memref.subview %arg0[3, 7, 2] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 95>>
    %collapse_shape_202 = memref.collapse_shape %subview_201 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 95>> into memref<f32, strided<[], offset: 95>>
    %405 = memref.load %collapse_shape_202[] : memref<f32, strided<[], offset: 95>>
    %subview_203 = memref.subview %arg0[3, 7, 1] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 94>>
    %collapse_shape_204 = memref.collapse_shape %subview_203 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 94>> into memref<f32, strided<[], offset: 94>>
    %406 = memref.load %collapse_shape_204[] : memref<f32, strided<[], offset: 94>>
    %subview_205 = memref.subview %arg0[3, 7, 0] [1, 1, 1] [1, 1, 1] : memref<4x8x3xf32> to memref<1x1x1xf32, strided<[24, 3, 1], offset: 93>>
    %collapse_shape_206 = memref.collapse_shape %subview_205 [] : memref<1x1x1xf32, strided<[24, 3, 1], offset: 93>> into memref<f32, strided<[], offset: 93>>
    %407 = memref.load %collapse_shape_206[] : memref<f32, strided<[], offset: 93>>
    %408 = arith.extf %407 : f32 to f64
    %409 = memref.load %alloca[] : memref<index>
    memref.store %408, %alloc[%409] : memref<?xf64>
    %410 = index.add %409, %idx1
    memref.store %410, %alloca[] : memref<index>
    %411 = arith.extf %406 : f32 to f64
    %412 = memref.load %alloca[] : memref<index>
    memref.store %411, %alloc[%412] : memref<?xf64>
    %413 = index.add %412, %idx1
    memref.store %413, %alloca[] : memref<index>
    %414 = arith.extf %405 : f32 to f64
    %415 = memref.load %alloca[] : memref<index>
    memref.store %414, %alloc[%415] : memref<?xf64>
    %416 = index.add %415, %idx1
    memref.store %416, %alloca[] : memref<index>
    %417 = call @qnode_forward_0.quantum(%arg0, %arg1, %alloc) : (memref<4x8x3xf32>, memref<8xf32>, memref<?xf64>) -> memref<f64>
    return %417 : memref<f64>
  }
  func.func @setup() {
    quantum.init
    return
  }
  func.func @teardown() {
    quantum.finalize
    return
  }
}