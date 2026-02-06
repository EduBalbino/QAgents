module @qnode_forward {
  memref.global "private" constant @__constant_xf32 : memref<f32> = dense<3.14159274> {alignment = 64 : i64}
  func.func public @jit_qnode_forward(%arg0: memref<2x4x3xf32>, %arg1: memref<4xf32>) -> memref<f64> attributes {llvm.copy_memref, llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(3735928559 : index) : i64
    %1 = call @qnode_forward_0(%arg0, %arg1) : (memref<2x4x3xf32>, memref<4xf32>) -> memref<f64>
    %2 = builtin.unrealized_conversion_cast %1 : memref<f64> to !llvm.struct<(ptr, ptr, i64)>
    %3 = llvm.extractvalue %2[0] : !llvm.struct<(ptr, ptr, i64)> 
    %4 = llvm.ptrtoint %3 : !llvm.ptr to i64
    %5 = llvm.icmp "eq" %0, %4 : i64
    %6 = scf.if %5 -> (memref<f64>) {
      %alloc = memref.alloc() : memref<f64>
      memref.copy %1, %alloc : memref<f64> to memref<f64>
      scf.yield %alloc : memref<f64>
    } else {
      scf.yield %1 : memref<f64>
    }
    return %6 : memref<f64>
  }
  func.func public @qnode_forward_0(%arg0: memref<2x4x3xf32>, %arg1: memref<4xf32>) -> memref<f64> attributes {diff_method = "adjoint", llvm.linkage = #llvm.linkage<internal>, qnode} {
    %c0_i64 = arith.constant 0 : i64
    %0 = memref.get_global @__constant_xf32 : memref<f32>
    %subview = memref.subview %arg0[1, 0, 2] [1, 1, 1] [1, 1, 1] : memref<2x4x3xf32> to memref<1x1x1xf32, strided<[12, 3, 1], offset: 14>>
    %collapse_shape = memref.collapse_shape %subview [] : memref<1x1x1xf32, strided<[12, 3, 1], offset: 14>> into memref<f32, strided<[], offset: 14>>
    %1 = memref.load %collapse_shape[] : memref<f32, strided<[], offset: 14>>
    %subview_0 = memref.subview %arg0[1, 0, 1] [1, 1, 1] [1, 1, 1] : memref<2x4x3xf32> to memref<1x1x1xf32, strided<[12, 3, 1], offset: 13>>
    %collapse_shape_1 = memref.collapse_shape %subview_0 [] : memref<1x1x1xf32, strided<[12, 3, 1], offset: 13>> into memref<f32, strided<[], offset: 13>>
    %2 = memref.load %collapse_shape_1[] : memref<f32, strided<[], offset: 13>>
    %subview_2 = memref.subview %arg0[1, 0, 0] [1, 1, 1] [1, 1, 1] : memref<2x4x3xf32> to memref<1x1x1xf32, strided<[12, 3, 1], offset: 12>>
    %collapse_shape_3 = memref.collapse_shape %subview_2 [] : memref<1x1x1xf32, strided<[12, 3, 1], offset: 12>> into memref<f32, strided<[], offset: 12>>
    %3 = memref.load %collapse_shape_3[] : memref<f32, strided<[], offset: 12>>
    %subview_4 = memref.subview %arg0[0, 3, 2] [1, 1, 1] [1, 1, 1] : memref<2x4x3xf32> to memref<1x1x1xf32, strided<[12, 3, 1], offset: 11>>
    %collapse_shape_5 = memref.collapse_shape %subview_4 [] : memref<1x1x1xf32, strided<[12, 3, 1], offset: 11>> into memref<f32, strided<[], offset: 11>>
    %4 = memref.load %collapse_shape_5[] : memref<f32, strided<[], offset: 11>>
    %subview_6 = memref.subview %arg0[0, 3, 1] [1, 1, 1] [1, 1, 1] : memref<2x4x3xf32> to memref<1x1x1xf32, strided<[12, 3, 1], offset: 10>>
    %collapse_shape_7 = memref.collapse_shape %subview_6 [] : memref<1x1x1xf32, strided<[12, 3, 1], offset: 10>> into memref<f32, strided<[], offset: 10>>
    %5 = memref.load %collapse_shape_7[] : memref<f32, strided<[], offset: 10>>
    %subview_8 = memref.subview %arg0[0, 3, 0] [1, 1, 1] [1, 1, 1] : memref<2x4x3xf32> to memref<1x1x1xf32, strided<[12, 3, 1], offset: 9>>
    %collapse_shape_9 = memref.collapse_shape %subview_8 [] : memref<1x1x1xf32, strided<[12, 3, 1], offset: 9>> into memref<f32, strided<[], offset: 9>>
    %6 = memref.load %collapse_shape_9[] : memref<f32, strided<[], offset: 9>>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<4xf32>
    linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%0 : memref<f32>) outs(%alloc : memref<4xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%alloc, %arg1 : memref<4xf32>, memref<4xf32>) outs(%alloc : memref<4xf32>) {
    ^bb0(%in: f32, %in_90: f32, %out: f32):
      %68 = arith.mulf %in, %in_90 : f32
      linalg.yield %68 : f32
    }
    %subview_10 = memref.subview %alloc[3] [1] [1] : memref<4xf32> to memref<1xf32, strided<[1], offset: 3>>
    %collapse_shape_11 = memref.collapse_shape %subview_10 [] : memref<1xf32, strided<[1], offset: 3>> into memref<f32, strided<[], offset: 3>>
    %7 = memref.load %collapse_shape_11[] : memref<f32, strided<[], offset: 3>>
    quantum.device shots(%c0_i64) ["/home/pichau/QAgents/.venv/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.so", "LightningSimulator", "{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}"]
    %8 = quantum.alloc( 4) : !quantum.reg
    %9 = quantum.extract %8[ 3] : !quantum.reg -> !quantum.bit
    %10 = arith.extf %7 : f32 to f64
    %out_qubits = quantum.custom "RY"(%10) %9 : !quantum.bit
    %11 = arith.extf %6 : f32 to f64
    %out_qubits_12 = quantum.custom "RZ"(%11) %out_qubits : !quantum.bit
    %12 = arith.extf %5 : f32 to f64
    %out_qubits_13 = quantum.custom "RY"(%12) %out_qubits_12 : !quantum.bit
    %13 = arith.extf %4 : f32 to f64
    %out_qubits_14 = quantum.custom "RZ"(%13) %out_qubits_13 : !quantum.bit
    %subview_15 = memref.subview %arg0[0, 2, 2] [1, 1, 1] [1, 1, 1] : memref<2x4x3xf32> to memref<1x1x1xf32, strided<[12, 3, 1], offset: 8>>
    %collapse_shape_16 = memref.collapse_shape %subview_15 [] : memref<1x1x1xf32, strided<[12, 3, 1], offset: 8>> into memref<f32, strided<[], offset: 8>>
    %14 = memref.load %collapse_shape_16[] : memref<f32, strided<[], offset: 8>>
    %subview_17 = memref.subview %arg0[0, 2, 1] [1, 1, 1] [1, 1, 1] : memref<2x4x3xf32> to memref<1x1x1xf32, strided<[12, 3, 1], offset: 7>>
    %collapse_shape_18 = memref.collapse_shape %subview_17 [] : memref<1x1x1xf32, strided<[12, 3, 1], offset: 7>> into memref<f32, strided<[], offset: 7>>
    %15 = memref.load %collapse_shape_18[] : memref<f32, strided<[], offset: 7>>
    %subview_19 = memref.subview %arg0[0, 2, 0] [1, 1, 1] [1, 1, 1] : memref<2x4x3xf32> to memref<1x1x1xf32, strided<[12, 3, 1], offset: 6>>
    %collapse_shape_20 = memref.collapse_shape %subview_19 [] : memref<1x1x1xf32, strided<[12, 3, 1], offset: 6>> into memref<f32, strided<[], offset: 6>>
    %16 = memref.load %collapse_shape_20[] : memref<f32, strided<[], offset: 6>>
    %subview_21 = memref.subview %alloc[2] [1] [1] : memref<4xf32> to memref<1xf32, strided<[1], offset: 2>>
    %collapse_shape_22 = memref.collapse_shape %subview_21 [] : memref<1xf32, strided<[1], offset: 2>> into memref<f32, strided<[], offset: 2>>
    %17 = memref.load %collapse_shape_22[] : memref<f32, strided<[], offset: 2>>
    %18 = quantum.extract %8[ 2] : !quantum.reg -> !quantum.bit
    %19 = arith.extf %17 : f32 to f64
    %out_qubits_23 = quantum.custom "RY"(%19) %18 : !quantum.bit
    %20 = arith.extf %16 : f32 to f64
    %out_qubits_24 = quantum.custom "RZ"(%20) %out_qubits_23 : !quantum.bit
    %21 = arith.extf %15 : f32 to f64
    %out_qubits_25 = quantum.custom "RY"(%21) %out_qubits_24 : !quantum.bit
    %22 = arith.extf %14 : f32 to f64
    %out_qubits_26 = quantum.custom "RZ"(%22) %out_qubits_25 : !quantum.bit
    %subview_27 = memref.subview %arg0[0, 0, 2] [1, 1, 1] [1, 1, 1] : memref<2x4x3xf32> to memref<1x1x1xf32, strided<[12, 3, 1], offset: 2>>
    %collapse_shape_28 = memref.collapse_shape %subview_27 [] : memref<1x1x1xf32, strided<[12, 3, 1], offset: 2>> into memref<f32, strided<[], offset: 2>>
    %23 = memref.load %collapse_shape_28[] : memref<f32, strided<[], offset: 2>>
    %subview_29 = memref.subview %arg0[0, 0, 1] [1, 1, 1] [1, 1, 1] : memref<2x4x3xf32> to memref<1x1x1xf32, strided<[12, 3, 1], offset: 1>>
    %collapse_shape_30 = memref.collapse_shape %subview_29 [] : memref<1x1x1xf32, strided<[12, 3, 1], offset: 1>> into memref<f32, strided<[], offset: 1>>
    %24 = memref.load %collapse_shape_30[] : memref<f32, strided<[], offset: 1>>
    %subview_31 = memref.subview %arg0[0, 0, 0] [1, 1, 1] [1, 1, 1] : memref<2x4x3xf32> to memref<1x1x1xf32, strided<[12, 3, 1]>>
    %collapse_shape_32 = memref.collapse_shape %subview_31 [] : memref<1x1x1xf32, strided<[12, 3, 1]>> into memref<f32, strided<[]>>
    %25 = memref.load %collapse_shape_32[] : memref<f32, strided<[]>>
    %subview_33 = memref.subview %alloc[0] [1] [1] : memref<4xf32> to memref<1xf32, strided<[1]>>
    %collapse_shape_34 = memref.collapse_shape %subview_33 [] : memref<1xf32, strided<[1]>> into memref<f32>
    %26 = memref.load %collapse_shape_34[] : memref<f32>
    %27 = quantum.extract %8[ 0] : !quantum.reg -> !quantum.bit
    %28 = arith.extf %26 : f32 to f64
    %out_qubits_35 = quantum.custom "RY"(%28) %27 : !quantum.bit
    %29 = arith.extf %25 : f32 to f64
    %out_qubits_36 = quantum.custom "RZ"(%29) %out_qubits_35 : !quantum.bit
    %30 = arith.extf %24 : f32 to f64
    %out_qubits_37 = quantum.custom "RY"(%30) %out_qubits_36 : !quantum.bit
    %31 = arith.extf %23 : f32 to f64
    %out_qubits_38 = quantum.custom "RZ"(%31) %out_qubits_37 : !quantum.bit
    %subview_39 = memref.subview %arg0[0, 1, 2] [1, 1, 1] [1, 1, 1] : memref<2x4x3xf32> to memref<1x1x1xf32, strided<[12, 3, 1], offset: 5>>
    %collapse_shape_40 = memref.collapse_shape %subview_39 [] : memref<1x1x1xf32, strided<[12, 3, 1], offset: 5>> into memref<f32, strided<[], offset: 5>>
    %32 = memref.load %collapse_shape_40[] : memref<f32, strided<[], offset: 5>>
    %subview_41 = memref.subview %arg0[0, 1, 1] [1, 1, 1] [1, 1, 1] : memref<2x4x3xf32> to memref<1x1x1xf32, strided<[12, 3, 1], offset: 4>>
    %collapse_shape_42 = memref.collapse_shape %subview_41 [] : memref<1x1x1xf32, strided<[12, 3, 1], offset: 4>> into memref<f32, strided<[], offset: 4>>
    %33 = memref.load %collapse_shape_42[] : memref<f32, strided<[], offset: 4>>
    %subview_43 = memref.subview %arg0[0, 1, 0] [1, 1, 1] [1, 1, 1] : memref<2x4x3xf32> to memref<1x1x1xf32, strided<[12, 3, 1], offset: 3>>
    %collapse_shape_44 = memref.collapse_shape %subview_43 [] : memref<1x1x1xf32, strided<[12, 3, 1], offset: 3>> into memref<f32, strided<[], offset: 3>>
    %34 = memref.load %collapse_shape_44[] : memref<f32, strided<[], offset: 3>>
    %subview_45 = memref.subview %alloc[1] [1] [1] : memref<4xf32> to memref<1xf32, strided<[1], offset: 1>>
    %collapse_shape_46 = memref.collapse_shape %subview_45 [] : memref<1xf32, strided<[1], offset: 1>> into memref<f32, strided<[], offset: 1>>
    %35 = memref.load %collapse_shape_46[] : memref<f32, strided<[], offset: 1>>
    memref.dealloc %alloc : memref<4xf32>
    %36 = quantum.extract %8[ 1] : !quantum.reg -> !quantum.bit
    %37 = arith.extf %35 : f32 to f64
    %out_qubits_47 = quantum.custom "RY"(%37) %36 : !quantum.bit
    %38 = arith.extf %34 : f32 to f64
    %out_qubits_48 = quantum.custom "RZ"(%38) %out_qubits_47 : !quantum.bit
    %39 = arith.extf %33 : f32 to f64
    %out_qubits_49 = quantum.custom "RY"(%39) %out_qubits_48 : !quantum.bit
    %40 = arith.extf %32 : f32 to f64
    %out_qubits_50 = quantum.custom "RZ"(%40) %out_qubits_49 : !quantum.bit
    %out_qubits_51:2 = quantum.custom "CNOT"() %out_qubits_38, %out_qubits_50 : !quantum.bit, !quantum.bit
    %out_qubits_52:2 = quantum.custom "CNOT"() %out_qubits_51#1, %out_qubits_26 : !quantum.bit, !quantum.bit
    %out_qubits_53:2 = quantum.custom "CNOT"() %out_qubits_52#1, %out_qubits_14 : !quantum.bit, !quantum.bit
    %out_qubits_54:2 = quantum.custom "CNOT"() %out_qubits_53#1, %out_qubits_51#0 : !quantum.bit, !quantum.bit
    %41 = arith.extf %3 : f32 to f64
    %out_qubits_55 = quantum.custom "RZ"(%41) %out_qubits_54#1 : !quantum.bit
    %42 = arith.extf %2 : f32 to f64
    %out_qubits_56 = quantum.custom "RY"(%42) %out_qubits_55 : !quantum.bit
    %43 = arith.extf %1 : f32 to f64
    %out_qubits_57 = quantum.custom "RZ"(%43) %out_qubits_56 : !quantum.bit
    %subview_58 = memref.subview %arg0[1, 2, 2] [1, 1, 1] [1, 1, 1] : memref<2x4x3xf32> to memref<1x1x1xf32, strided<[12, 3, 1], offset: 20>>
    %collapse_shape_59 = memref.collapse_shape %subview_58 [] : memref<1x1x1xf32, strided<[12, 3, 1], offset: 20>> into memref<f32, strided<[], offset: 20>>
    %44 = memref.load %collapse_shape_59[] : memref<f32, strided<[], offset: 20>>
    %subview_60 = memref.subview %arg0[1, 2, 1] [1, 1, 1] [1, 1, 1] : memref<2x4x3xf32> to memref<1x1x1xf32, strided<[12, 3, 1], offset: 19>>
    %collapse_shape_61 = memref.collapse_shape %subview_60 [] : memref<1x1x1xf32, strided<[12, 3, 1], offset: 19>> into memref<f32, strided<[], offset: 19>>
    %45 = memref.load %collapse_shape_61[] : memref<f32, strided<[], offset: 19>>
    %subview_62 = memref.subview %arg0[1, 2, 0] [1, 1, 1] [1, 1, 1] : memref<2x4x3xf32> to memref<1x1x1xf32, strided<[12, 3, 1], offset: 18>>
    %collapse_shape_63 = memref.collapse_shape %subview_62 [] : memref<1x1x1xf32, strided<[12, 3, 1], offset: 18>> into memref<f32, strided<[], offset: 18>>
    %46 = memref.load %collapse_shape_63[] : memref<f32, strided<[], offset: 18>>
    %47 = arith.extf %46 : f32 to f64
    %out_qubits_64 = quantum.custom "RZ"(%47) %out_qubits_53#0 : !quantum.bit
    %48 = arith.extf %45 : f32 to f64
    %out_qubits_65 = quantum.custom "RY"(%48) %out_qubits_64 : !quantum.bit
    %49 = arith.extf %44 : f32 to f64
    %out_qubits_66 = quantum.custom "RZ"(%49) %out_qubits_65 : !quantum.bit
    %out_qubits_67:2 = quantum.custom "CNOT"() %out_qubits_57, %out_qubits_66 : !quantum.bit, !quantum.bit
    %out_qubits_68:2 = quantum.custom "CNOT"() %out_qubits_67#1, %out_qubits_67#0 : !quantum.bit, !quantum.bit
    %subview_69 = memref.subview %arg0[1, 1, 2] [1, 1, 1] [1, 1, 1] : memref<2x4x3xf32> to memref<1x1x1xf32, strided<[12, 3, 1], offset: 17>>
    %collapse_shape_70 = memref.collapse_shape %subview_69 [] : memref<1x1x1xf32, strided<[12, 3, 1], offset: 17>> into memref<f32, strided<[], offset: 17>>
    %50 = memref.load %collapse_shape_70[] : memref<f32, strided<[], offset: 17>>
    %subview_71 = memref.subview %arg0[1, 1, 1] [1, 1, 1] [1, 1, 1] : memref<2x4x3xf32> to memref<1x1x1xf32, strided<[12, 3, 1], offset: 16>>
    %collapse_shape_72 = memref.collapse_shape %subview_71 [] : memref<1x1x1xf32, strided<[12, 3, 1], offset: 16>> into memref<f32, strided<[], offset: 16>>
    %51 = memref.load %collapse_shape_72[] : memref<f32, strided<[], offset: 16>>
    %subview_73 = memref.subview %arg0[1, 1, 0] [1, 1, 1] [1, 1, 1] : memref<2x4x3xf32> to memref<1x1x1xf32, strided<[12, 3, 1], offset: 15>>
    %collapse_shape_74 = memref.collapse_shape %subview_73 [] : memref<1x1x1xf32, strided<[12, 3, 1], offset: 15>> into memref<f32, strided<[], offset: 15>>
    %52 = memref.load %collapse_shape_74[] : memref<f32, strided<[], offset: 15>>
    %53 = arith.extf %52 : f32 to f64
    %out_qubits_75 = quantum.custom "RZ"(%53) %out_qubits_52#0 : !quantum.bit
    %54 = arith.extf %51 : f32 to f64
    %out_qubits_76 = quantum.custom "RY"(%54) %out_qubits_75 : !quantum.bit
    %55 = arith.extf %50 : f32 to f64
    %out_qubits_77 = quantum.custom "RZ"(%55) %out_qubits_76 : !quantum.bit
    %subview_78 = memref.subview %arg0[1, 3, 2] [1, 1, 1] [1, 1, 1] : memref<2x4x3xf32> to memref<1x1x1xf32, strided<[12, 3, 1], offset: 23>>
    %collapse_shape_79 = memref.collapse_shape %subview_78 [] : memref<1x1x1xf32, strided<[12, 3, 1], offset: 23>> into memref<f32, strided<[], offset: 23>>
    %56 = memref.load %collapse_shape_79[] : memref<f32, strided<[], offset: 23>>
    %subview_80 = memref.subview %arg0[1, 3, 1] [1, 1, 1] [1, 1, 1] : memref<2x4x3xf32> to memref<1x1x1xf32, strided<[12, 3, 1], offset: 22>>
    %collapse_shape_81 = memref.collapse_shape %subview_80 [] : memref<1x1x1xf32, strided<[12, 3, 1], offset: 22>> into memref<f32, strided<[], offset: 22>>
    %57 = memref.load %collapse_shape_81[] : memref<f32, strided<[], offset: 22>>
    %subview_82 = memref.subview %arg0[1, 3, 0] [1, 1, 1] [1, 1, 1] : memref<2x4x3xf32> to memref<1x1x1xf32, strided<[12, 3, 1], offset: 21>>
    %collapse_shape_83 = memref.collapse_shape %subview_82 [] : memref<1x1x1xf32, strided<[12, 3, 1], offset: 21>> into memref<f32, strided<[], offset: 21>>
    %58 = memref.load %collapse_shape_83[] : memref<f32, strided<[], offset: 21>>
    %59 = arith.extf %58 : f32 to f64
    %out_qubits_84 = quantum.custom "RZ"(%59) %out_qubits_54#0 : !quantum.bit
    %60 = arith.extf %57 : f32 to f64
    %out_qubits_85 = quantum.custom "RY"(%60) %out_qubits_84 : !quantum.bit
    %61 = arith.extf %56 : f32 to f64
    %out_qubits_86 = quantum.custom "RZ"(%61) %out_qubits_85 : !quantum.bit
    %out_qubits_87:2 = quantum.custom "CNOT"() %out_qubits_77, %out_qubits_86 : !quantum.bit, !quantum.bit
    %out_qubits_88:2 = quantum.custom "CNOT"() %out_qubits_87#1, %out_qubits_87#0 : !quantum.bit, !quantum.bit
    %62 = quantum.namedobs %out_qubits_68#1[ PauliZ] : !quantum.obs
    %63 = quantum.expval %62 : f64
    %alloc_89 = memref.alloc() {alignment = 64 : i64} : memref<f64>
    memref.store %63, %alloc_89[] : memref<f64>
    %64 = quantum.insert %8[ 0], %out_qubits_68#1 : !quantum.reg, !quantum.bit
    %65 = quantum.insert %64[ 1], %out_qubits_88#1 : !quantum.reg, !quantum.bit
    %66 = quantum.insert %65[ 2], %out_qubits_68#0 : !quantum.reg, !quantum.bit
    %67 = quantum.insert %66[ 3], %out_qubits_88#0 : !quantum.reg, !quantum.bit
    quantum.dealloc %67 : !quantum.reg
    quantum.device_release
    return %alloc_89 : memref<f64>
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