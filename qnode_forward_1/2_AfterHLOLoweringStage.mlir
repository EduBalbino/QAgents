module @qnode_forward {
  func.func public @jit_qnode_forward(%arg0: tensor<2x4x3xf32>, %arg1: tensor<4xf32>) -> tensor<f64> attributes {llvm.emit_c_interface} {
    %0 = call @qnode_forward_0(%arg0, %arg1) : (tensor<2x4x3xf32>, tensor<4xf32>) -> tensor<f64>
    return %0 : tensor<f64>
  }
  func.func public @qnode_forward_0(%arg0: tensor<2x4x3xf32>, %arg1: tensor<4xf32>) -> tensor<f64> attributes {diff_method = "adjoint", llvm.linkage = #llvm.linkage<internal>, qnode} {
    %cst = arith.constant dense<3.14159274> : tensor<f32>
    %c0_i64 = arith.constant 0 : i64
    %extracted_slice = tensor.extract_slice %arg0[1, 0, 2] [1, 1, 1] [1, 1, 1] : tensor<2x4x3xf32> to tensor<1x1x1xf32>
    %collapsed = tensor.collapse_shape %extracted_slice [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted = tensor.extract %collapsed[] : tensor<f32>
    %extracted_slice_0 = tensor.extract_slice %arg0[1, 0, 1] [1, 1, 1] [1, 1, 1] : tensor<2x4x3xf32> to tensor<1x1x1xf32>
    %collapsed_1 = tensor.collapse_shape %extracted_slice_0 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_2 = tensor.extract %collapsed_1[] : tensor<f32>
    %extracted_slice_3 = tensor.extract_slice %arg0[1, 0, 0] [1, 1, 1] [1, 1, 1] : tensor<2x4x3xf32> to tensor<1x1x1xf32>
    %collapsed_4 = tensor.collapse_shape %extracted_slice_3 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_5 = tensor.extract %collapsed_4[] : tensor<f32>
    %extracted_slice_6 = tensor.extract_slice %arg0[0, 3, 2] [1, 1, 1] [1, 1, 1] : tensor<2x4x3xf32> to tensor<1x1x1xf32>
    %collapsed_7 = tensor.collapse_shape %extracted_slice_6 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_8 = tensor.extract %collapsed_7[] : tensor<f32>
    %extracted_slice_9 = tensor.extract_slice %arg0[0, 3, 1] [1, 1, 1] [1, 1, 1] : tensor<2x4x3xf32> to tensor<1x1x1xf32>
    %collapsed_10 = tensor.collapse_shape %extracted_slice_9 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_11 = tensor.extract %collapsed_10[] : tensor<f32>
    %extracted_slice_12 = tensor.extract_slice %arg0[0, 3, 0] [1, 1, 1] [1, 1, 1] : tensor<2x4x3xf32> to tensor<1x1x1xf32>
    %collapsed_13 = tensor.collapse_shape %extracted_slice_12 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_14 = tensor.extract %collapsed_13[] : tensor<f32>
    %0 = tensor.empty() : tensor<4xf32>
    %1 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%cst : tensor<f32>) outs(%0 : tensor<4xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4xf32>
    %2 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1, %arg1 : tensor<4xf32>, tensor<4xf32>) outs(%0 : tensor<4xf32>) {
    ^bb0(%in: f32, %in_116: f32, %out: f32):
      %42 = arith.mulf %in, %in_116 : f32
      linalg.yield %42 : f32
    } -> tensor<4xf32>
    %extracted_slice_15 = tensor.extract_slice %2[3] [1] [1] : tensor<4xf32> to tensor<1xf32>
    %collapsed_16 = tensor.collapse_shape %extracted_slice_15 [] : tensor<1xf32> into tensor<f32>
    %extracted_17 = tensor.extract %collapsed_16[] : tensor<f32>
    quantum.device shots(%c0_i64) ["/home/pichau/QAgents/.venv/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.so", "LightningSimulator", "{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}"]
    %3 = quantum.alloc( 4) : !quantum.reg
    %4 = quantum.extract %3[ 3] : !quantum.reg -> !quantum.bit
    %5 = arith.extf %extracted_17 : f32 to f64
    %out_qubits = quantum.custom "RY"(%5) %4 : !quantum.bit
    %6 = arith.extf %extracted_14 : f32 to f64
    %out_qubits_18 = quantum.custom "RZ"(%6) %out_qubits : !quantum.bit
    %7 = arith.extf %extracted_11 : f32 to f64
    %out_qubits_19 = quantum.custom "RY"(%7) %out_qubits_18 : !quantum.bit
    %8 = arith.extf %extracted_8 : f32 to f64
    %out_qubits_20 = quantum.custom "RZ"(%8) %out_qubits_19 : !quantum.bit
    %extracted_slice_21 = tensor.extract_slice %arg0[0, 2, 2] [1, 1, 1] [1, 1, 1] : tensor<2x4x3xf32> to tensor<1x1x1xf32>
    %collapsed_22 = tensor.collapse_shape %extracted_slice_21 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_23 = tensor.extract %collapsed_22[] : tensor<f32>
    %extracted_slice_24 = tensor.extract_slice %arg0[0, 2, 1] [1, 1, 1] [1, 1, 1] : tensor<2x4x3xf32> to tensor<1x1x1xf32>
    %collapsed_25 = tensor.collapse_shape %extracted_slice_24 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_26 = tensor.extract %collapsed_25[] : tensor<f32>
    %extracted_slice_27 = tensor.extract_slice %arg0[0, 2, 0] [1, 1, 1] [1, 1, 1] : tensor<2x4x3xf32> to tensor<1x1x1xf32>
    %collapsed_28 = tensor.collapse_shape %extracted_slice_27 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_29 = tensor.extract %collapsed_28[] : tensor<f32>
    %extracted_slice_30 = tensor.extract_slice %2[2] [1] [1] : tensor<4xf32> to tensor<1xf32>
    %collapsed_31 = tensor.collapse_shape %extracted_slice_30 [] : tensor<1xf32> into tensor<f32>
    %extracted_32 = tensor.extract %collapsed_31[] : tensor<f32>
    %9 = quantum.extract %3[ 2] : !quantum.reg -> !quantum.bit
    %10 = arith.extf %extracted_32 : f32 to f64
    %out_qubits_33 = quantum.custom "RY"(%10) %9 : !quantum.bit
    %11 = arith.extf %extracted_29 : f32 to f64
    %out_qubits_34 = quantum.custom "RZ"(%11) %out_qubits_33 : !quantum.bit
    %12 = arith.extf %extracted_26 : f32 to f64
    %out_qubits_35 = quantum.custom "RY"(%12) %out_qubits_34 : !quantum.bit
    %13 = arith.extf %extracted_23 : f32 to f64
    %out_qubits_36 = quantum.custom "RZ"(%13) %out_qubits_35 : !quantum.bit
    %extracted_slice_37 = tensor.extract_slice %arg0[0, 0, 2] [1, 1, 1] [1, 1, 1] : tensor<2x4x3xf32> to tensor<1x1x1xf32>
    %collapsed_38 = tensor.collapse_shape %extracted_slice_37 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_39 = tensor.extract %collapsed_38[] : tensor<f32>
    %extracted_slice_40 = tensor.extract_slice %arg0[0, 0, 1] [1, 1, 1] [1, 1, 1] : tensor<2x4x3xf32> to tensor<1x1x1xf32>
    %collapsed_41 = tensor.collapse_shape %extracted_slice_40 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_42 = tensor.extract %collapsed_41[] : tensor<f32>
    %extracted_slice_43 = tensor.extract_slice %arg0[0, 0, 0] [1, 1, 1] [1, 1, 1] : tensor<2x4x3xf32> to tensor<1x1x1xf32>
    %collapsed_44 = tensor.collapse_shape %extracted_slice_43 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_45 = tensor.extract %collapsed_44[] : tensor<f32>
    %extracted_slice_46 = tensor.extract_slice %2[0] [1] [1] : tensor<4xf32> to tensor<1xf32>
    %collapsed_47 = tensor.collapse_shape %extracted_slice_46 [] : tensor<1xf32> into tensor<f32>
    %extracted_48 = tensor.extract %collapsed_47[] : tensor<f32>
    %14 = quantum.extract %3[ 0] : !quantum.reg -> !quantum.bit
    %15 = arith.extf %extracted_48 : f32 to f64
    %out_qubits_49 = quantum.custom "RY"(%15) %14 : !quantum.bit
    %16 = arith.extf %extracted_45 : f32 to f64
    %out_qubits_50 = quantum.custom "RZ"(%16) %out_qubits_49 : !quantum.bit
    %17 = arith.extf %extracted_42 : f32 to f64
    %out_qubits_51 = quantum.custom "RY"(%17) %out_qubits_50 : !quantum.bit
    %18 = arith.extf %extracted_39 : f32 to f64
    %out_qubits_52 = quantum.custom "RZ"(%18) %out_qubits_51 : !quantum.bit
    %extracted_slice_53 = tensor.extract_slice %arg0[0, 1, 2] [1, 1, 1] [1, 1, 1] : tensor<2x4x3xf32> to tensor<1x1x1xf32>
    %collapsed_54 = tensor.collapse_shape %extracted_slice_53 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_55 = tensor.extract %collapsed_54[] : tensor<f32>
    %extracted_slice_56 = tensor.extract_slice %arg0[0, 1, 1] [1, 1, 1] [1, 1, 1] : tensor<2x4x3xf32> to tensor<1x1x1xf32>
    %collapsed_57 = tensor.collapse_shape %extracted_slice_56 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_58 = tensor.extract %collapsed_57[] : tensor<f32>
    %extracted_slice_59 = tensor.extract_slice %arg0[0, 1, 0] [1, 1, 1] [1, 1, 1] : tensor<2x4x3xf32> to tensor<1x1x1xf32>
    %collapsed_60 = tensor.collapse_shape %extracted_slice_59 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_61 = tensor.extract %collapsed_60[] : tensor<f32>
    %extracted_slice_62 = tensor.extract_slice %2[1] [1] [1] : tensor<4xf32> to tensor<1xf32>
    %collapsed_63 = tensor.collapse_shape %extracted_slice_62 [] : tensor<1xf32> into tensor<f32>
    %extracted_64 = tensor.extract %collapsed_63[] : tensor<f32>
    %19 = quantum.extract %3[ 1] : !quantum.reg -> !quantum.bit
    %20 = arith.extf %extracted_64 : f32 to f64
    %out_qubits_65 = quantum.custom "RY"(%20) %19 : !quantum.bit
    %21 = arith.extf %extracted_61 : f32 to f64
    %out_qubits_66 = quantum.custom "RZ"(%21) %out_qubits_65 : !quantum.bit
    %22 = arith.extf %extracted_58 : f32 to f64
    %out_qubits_67 = quantum.custom "RY"(%22) %out_qubits_66 : !quantum.bit
    %23 = arith.extf %extracted_55 : f32 to f64
    %out_qubits_68 = quantum.custom "RZ"(%23) %out_qubits_67 : !quantum.bit
    %out_qubits_69:2 = quantum.custom "CNOT"() %out_qubits_52, %out_qubits_68 : !quantum.bit, !quantum.bit
    %out_qubits_70:2 = quantum.custom "CNOT"() %out_qubits_69#1, %out_qubits_36 : !quantum.bit, !quantum.bit
    %out_qubits_71:2 = quantum.custom "CNOT"() %out_qubits_70#1, %out_qubits_20 : !quantum.bit, !quantum.bit
    %out_qubits_72:2 = quantum.custom "CNOT"() %out_qubits_71#1, %out_qubits_69#0 : !quantum.bit, !quantum.bit
    %24 = arith.extf %extracted_5 : f32 to f64
    %out_qubits_73 = quantum.custom "RZ"(%24) %out_qubits_72#1 : !quantum.bit
    %25 = arith.extf %extracted_2 : f32 to f64
    %out_qubits_74 = quantum.custom "RY"(%25) %out_qubits_73 : !quantum.bit
    %26 = arith.extf %extracted : f32 to f64
    %out_qubits_75 = quantum.custom "RZ"(%26) %out_qubits_74 : !quantum.bit
    %extracted_slice_76 = tensor.extract_slice %arg0[1, 2, 2] [1, 1, 1] [1, 1, 1] : tensor<2x4x3xf32> to tensor<1x1x1xf32>
    %collapsed_77 = tensor.collapse_shape %extracted_slice_76 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_78 = tensor.extract %collapsed_77[] : tensor<f32>
    %extracted_slice_79 = tensor.extract_slice %arg0[1, 2, 1] [1, 1, 1] [1, 1, 1] : tensor<2x4x3xf32> to tensor<1x1x1xf32>
    %collapsed_80 = tensor.collapse_shape %extracted_slice_79 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_81 = tensor.extract %collapsed_80[] : tensor<f32>
    %extracted_slice_82 = tensor.extract_slice %arg0[1, 2, 0] [1, 1, 1] [1, 1, 1] : tensor<2x4x3xf32> to tensor<1x1x1xf32>
    %collapsed_83 = tensor.collapse_shape %extracted_slice_82 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_84 = tensor.extract %collapsed_83[] : tensor<f32>
    %27 = arith.extf %extracted_84 : f32 to f64
    %out_qubits_85 = quantum.custom "RZ"(%27) %out_qubits_71#0 : !quantum.bit
    %28 = arith.extf %extracted_81 : f32 to f64
    %out_qubits_86 = quantum.custom "RY"(%28) %out_qubits_85 : !quantum.bit
    %29 = arith.extf %extracted_78 : f32 to f64
    %out_qubits_87 = quantum.custom "RZ"(%29) %out_qubits_86 : !quantum.bit
    %out_qubits_88:2 = quantum.custom "CNOT"() %out_qubits_75, %out_qubits_87 : !quantum.bit, !quantum.bit
    %out_qubits_89:2 = quantum.custom "CNOT"() %out_qubits_88#1, %out_qubits_88#0 : !quantum.bit, !quantum.bit
    %extracted_slice_90 = tensor.extract_slice %arg0[1, 1, 2] [1, 1, 1] [1, 1, 1] : tensor<2x4x3xf32> to tensor<1x1x1xf32>
    %collapsed_91 = tensor.collapse_shape %extracted_slice_90 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_92 = tensor.extract %collapsed_91[] : tensor<f32>
    %extracted_slice_93 = tensor.extract_slice %arg0[1, 1, 1] [1, 1, 1] [1, 1, 1] : tensor<2x4x3xf32> to tensor<1x1x1xf32>
    %collapsed_94 = tensor.collapse_shape %extracted_slice_93 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_95 = tensor.extract %collapsed_94[] : tensor<f32>
    %extracted_slice_96 = tensor.extract_slice %arg0[1, 1, 0] [1, 1, 1] [1, 1, 1] : tensor<2x4x3xf32> to tensor<1x1x1xf32>
    %collapsed_97 = tensor.collapse_shape %extracted_slice_96 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_98 = tensor.extract %collapsed_97[] : tensor<f32>
    %30 = arith.extf %extracted_98 : f32 to f64
    %out_qubits_99 = quantum.custom "RZ"(%30) %out_qubits_70#0 : !quantum.bit
    %31 = arith.extf %extracted_95 : f32 to f64
    %out_qubits_100 = quantum.custom "RY"(%31) %out_qubits_99 : !quantum.bit
    %32 = arith.extf %extracted_92 : f32 to f64
    %out_qubits_101 = quantum.custom "RZ"(%32) %out_qubits_100 : !quantum.bit
    %extracted_slice_102 = tensor.extract_slice %arg0[1, 3, 2] [1, 1, 1] [1, 1, 1] : tensor<2x4x3xf32> to tensor<1x1x1xf32>
    %collapsed_103 = tensor.collapse_shape %extracted_slice_102 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_104 = tensor.extract %collapsed_103[] : tensor<f32>
    %extracted_slice_105 = tensor.extract_slice %arg0[1, 3, 1] [1, 1, 1] [1, 1, 1] : tensor<2x4x3xf32> to tensor<1x1x1xf32>
    %collapsed_106 = tensor.collapse_shape %extracted_slice_105 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_107 = tensor.extract %collapsed_106[] : tensor<f32>
    %extracted_slice_108 = tensor.extract_slice %arg0[1, 3, 0] [1, 1, 1] [1, 1, 1] : tensor<2x4x3xf32> to tensor<1x1x1xf32>
    %collapsed_109 = tensor.collapse_shape %extracted_slice_108 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_110 = tensor.extract %collapsed_109[] : tensor<f32>
    %33 = arith.extf %extracted_110 : f32 to f64
    %out_qubits_111 = quantum.custom "RZ"(%33) %out_qubits_72#0 : !quantum.bit
    %34 = arith.extf %extracted_107 : f32 to f64
    %out_qubits_112 = quantum.custom "RY"(%34) %out_qubits_111 : !quantum.bit
    %35 = arith.extf %extracted_104 : f32 to f64
    %out_qubits_113 = quantum.custom "RZ"(%35) %out_qubits_112 : !quantum.bit
    %out_qubits_114:2 = quantum.custom "CNOT"() %out_qubits_101, %out_qubits_113 : !quantum.bit, !quantum.bit
    %out_qubits_115:2 = quantum.custom "CNOT"() %out_qubits_114#1, %out_qubits_114#0 : !quantum.bit, !quantum.bit
    %36 = quantum.namedobs %out_qubits_89#1[ PauliZ] : !quantum.obs
    %37 = quantum.expval %36 : f64
    %from_elements = tensor.from_elements %37 : tensor<f64>
    %38 = quantum.insert %3[ 0], %out_qubits_89#1 : !quantum.reg, !quantum.bit
    %39 = quantum.insert %38[ 1], %out_qubits_115#1 : !quantum.reg, !quantum.bit
    %40 = quantum.insert %39[ 2], %out_qubits_89#0 : !quantum.reg, !quantum.bit
    %41 = quantum.insert %40[ 3], %out_qubits_115#0 : !quantum.reg, !quantum.bit
    quantum.dealloc %41 : !quantum.reg
    quantum.device_release
    return %from_elements : tensor<f64>
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