module @deriv_qnode_forward {
  func.func public @jit_deriv_qnode_forward(%arg0: tensor<4x8x3xf32>, %arg1: tensor<8xf32>) -> tensor<4x8x3xf32> attributes {llvm.emit_c_interface} {
    %0 = call @qnode_forward_0.pcount(%arg0, %arg1) : (tensor<4x8x3xf32>, tensor<8xf32>) -> index
    %1 = call @qnode_forward_0.fullgrad0(%arg0, %arg1, %0) : (tensor<4x8x3xf32>, tensor<8xf32>, index) -> tensor<4x8x3xf32>
    return %1 : tensor<4x8x3xf32>
  }
  func.func public @qnode_forward_0(%arg0: tensor<4x8x3xf32>, %arg1: tensor<8xf32>) -> tensor<f64> attributes {diff_method = "adjoint", llvm.linkage = #llvm.linkage<internal>, qnode} {
    %cst = arith.constant dense<3.14159274> : tensor<f32>
    %c0_i64 = arith.constant 0 : i64
    %extracted_slice = tensor.extract_slice %arg0[3, 0, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed = tensor.collapse_shape %extracted_slice [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted = tensor.extract %collapsed[] : tensor<f32>
    %extracted_slice_0 = tensor.extract_slice %arg0[3, 0, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_1 = tensor.collapse_shape %extracted_slice_0 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_2 = tensor.extract %collapsed_1[] : tensor<f32>
    %extracted_slice_3 = tensor.extract_slice %arg0[3, 0, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_4 = tensor.collapse_shape %extracted_slice_3 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_5 = tensor.extract %collapsed_4[] : tensor<f32>
    %extracted_slice_6 = tensor.extract_slice %arg0[2, 0, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_7 = tensor.collapse_shape %extracted_slice_6 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_8 = tensor.extract %collapsed_7[] : tensor<f32>
    %extracted_slice_9 = tensor.extract_slice %arg0[2, 0, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_10 = tensor.collapse_shape %extracted_slice_9 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_11 = tensor.extract %collapsed_10[] : tensor<f32>
    %extracted_slice_12 = tensor.extract_slice %arg0[2, 0, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_13 = tensor.collapse_shape %extracted_slice_12 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_14 = tensor.extract %collapsed_13[] : tensor<f32>
    %extracted_slice_15 = tensor.extract_slice %arg0[1, 6, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_16 = tensor.collapse_shape %extracted_slice_15 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_17 = tensor.extract %collapsed_16[] : tensor<f32>
    %extracted_slice_18 = tensor.extract_slice %arg0[1, 6, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_19 = tensor.collapse_shape %extracted_slice_18 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_20 = tensor.extract %collapsed_19[] : tensor<f32>
    %extracted_slice_21 = tensor.extract_slice %arg0[1, 6, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_22 = tensor.collapse_shape %extracted_slice_21 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_23 = tensor.extract %collapsed_22[] : tensor<f32>
    %extracted_slice_24 = tensor.extract_slice %arg0[0, 7, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_25 = tensor.collapse_shape %extracted_slice_24 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_26 = tensor.extract %collapsed_25[] : tensor<f32>
    %extracted_slice_27 = tensor.extract_slice %arg0[0, 7, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_28 = tensor.collapse_shape %extracted_slice_27 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_29 = tensor.extract %collapsed_28[] : tensor<f32>
    %extracted_slice_30 = tensor.extract_slice %arg0[0, 7, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_31 = tensor.collapse_shape %extracted_slice_30 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_32 = tensor.extract %collapsed_31[] : tensor<f32>
    %0 = tensor.empty() : tensor<8xf32>
    %1 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%cst : tensor<f32>) outs(%0 : tensor<8xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<8xf32>
    %2 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1, %arg1 : tensor<8xf32>, tensor<8xf32>) outs(%0 : tensor<8xf32>) {
    ^bb0(%in: f32, %in_444: f32, %out: f32):
      %126 = arith.mulf %in, %in_444 : f32
      linalg.yield %126 : f32
    } -> tensor<8xf32>
    %extracted_slice_33 = tensor.extract_slice %2[7] [1] [1] : tensor<8xf32> to tensor<1xf32>
    %collapsed_34 = tensor.collapse_shape %extracted_slice_33 [] : tensor<1xf32> into tensor<f32>
    %extracted_35 = tensor.extract %collapsed_34[] : tensor<f32>
    quantum.device shots(%c0_i64) ["/home/pichau/QAgents/.venv/lib/python3.12/site-packages/pennylane_lightning/liblightning_gpu_catalyst.so", "LightningGPUSimulator", "{}"]
    %3 = quantum.alloc( 8) : !quantum.reg
    %4 = quantum.extract %3[ 7] : !quantum.reg -> !quantum.bit
    %5 = arith.extf %extracted_35 : f32 to f64
    %out_qubits = quantum.custom "RY"(%5) %4 : !quantum.bit
    %6 = arith.extf %extracted_32 : f32 to f64
    %out_qubits_36 = quantum.custom "RZ"(%6) %out_qubits : !quantum.bit
    %7 = arith.extf %extracted_29 : f32 to f64
    %out_qubits_37 = quantum.custom "RY"(%7) %out_qubits_36 : !quantum.bit
    %8 = arith.extf %extracted_26 : f32 to f64
    %out_qubits_38 = quantum.custom "RZ"(%8) %out_qubits_37 : !quantum.bit
    %extracted_slice_39 = tensor.extract_slice %arg0[0, 6, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_40 = tensor.collapse_shape %extracted_slice_39 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_41 = tensor.extract %collapsed_40[] : tensor<f32>
    %extracted_slice_42 = tensor.extract_slice %arg0[0, 6, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_43 = tensor.collapse_shape %extracted_slice_42 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_44 = tensor.extract %collapsed_43[] : tensor<f32>
    %extracted_slice_45 = tensor.extract_slice %arg0[0, 6, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_46 = tensor.collapse_shape %extracted_slice_45 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_47 = tensor.extract %collapsed_46[] : tensor<f32>
    %extracted_slice_48 = tensor.extract_slice %2[6] [1] [1] : tensor<8xf32> to tensor<1xf32>
    %collapsed_49 = tensor.collapse_shape %extracted_slice_48 [] : tensor<1xf32> into tensor<f32>
    %extracted_50 = tensor.extract %collapsed_49[] : tensor<f32>
    %9 = quantum.extract %3[ 6] : !quantum.reg -> !quantum.bit
    %10 = arith.extf %extracted_50 : f32 to f64
    %out_qubits_51 = quantum.custom "RY"(%10) %9 : !quantum.bit
    %11 = arith.extf %extracted_47 : f32 to f64
    %out_qubits_52 = quantum.custom "RZ"(%11) %out_qubits_51 : !quantum.bit
    %12 = arith.extf %extracted_44 : f32 to f64
    %out_qubits_53 = quantum.custom "RY"(%12) %out_qubits_52 : !quantum.bit
    %13 = arith.extf %extracted_41 : f32 to f64
    %out_qubits_54 = quantum.custom "RZ"(%13) %out_qubits_53 : !quantum.bit
    %extracted_slice_55 = tensor.extract_slice %arg0[0, 5, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_56 = tensor.collapse_shape %extracted_slice_55 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_57 = tensor.extract %collapsed_56[] : tensor<f32>
    %extracted_slice_58 = tensor.extract_slice %arg0[0, 5, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_59 = tensor.collapse_shape %extracted_slice_58 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_60 = tensor.extract %collapsed_59[] : tensor<f32>
    %extracted_slice_61 = tensor.extract_slice %arg0[0, 5, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_62 = tensor.collapse_shape %extracted_slice_61 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_63 = tensor.extract %collapsed_62[] : tensor<f32>
    %extracted_slice_64 = tensor.extract_slice %2[5] [1] [1] : tensor<8xf32> to tensor<1xf32>
    %collapsed_65 = tensor.collapse_shape %extracted_slice_64 [] : tensor<1xf32> into tensor<f32>
    %extracted_66 = tensor.extract %collapsed_65[] : tensor<f32>
    %14 = quantum.extract %3[ 5] : !quantum.reg -> !quantum.bit
    %15 = arith.extf %extracted_66 : f32 to f64
    %out_qubits_67 = quantum.custom "RY"(%15) %14 : !quantum.bit
    %16 = arith.extf %extracted_63 : f32 to f64
    %out_qubits_68 = quantum.custom "RZ"(%16) %out_qubits_67 : !quantum.bit
    %17 = arith.extf %extracted_60 : f32 to f64
    %out_qubits_69 = quantum.custom "RY"(%17) %out_qubits_68 : !quantum.bit
    %18 = arith.extf %extracted_57 : f32 to f64
    %out_qubits_70 = quantum.custom "RZ"(%18) %out_qubits_69 : !quantum.bit
    %extracted_slice_71 = tensor.extract_slice %arg0[0, 4, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_72 = tensor.collapse_shape %extracted_slice_71 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_73 = tensor.extract %collapsed_72[] : tensor<f32>
    %extracted_slice_74 = tensor.extract_slice %arg0[0, 4, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_75 = tensor.collapse_shape %extracted_slice_74 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_76 = tensor.extract %collapsed_75[] : tensor<f32>
    %extracted_slice_77 = tensor.extract_slice %arg0[0, 4, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_78 = tensor.collapse_shape %extracted_slice_77 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_79 = tensor.extract %collapsed_78[] : tensor<f32>
    %extracted_slice_80 = tensor.extract_slice %2[4] [1] [1] : tensor<8xf32> to tensor<1xf32>
    %collapsed_81 = tensor.collapse_shape %extracted_slice_80 [] : tensor<1xf32> into tensor<f32>
    %extracted_82 = tensor.extract %collapsed_81[] : tensor<f32>
    %19 = quantum.extract %3[ 4] : !quantum.reg -> !quantum.bit
    %20 = arith.extf %extracted_82 : f32 to f64
    %out_qubits_83 = quantum.custom "RY"(%20) %19 : !quantum.bit
    %21 = arith.extf %extracted_79 : f32 to f64
    %out_qubits_84 = quantum.custom "RZ"(%21) %out_qubits_83 : !quantum.bit
    %22 = arith.extf %extracted_76 : f32 to f64
    %out_qubits_85 = quantum.custom "RY"(%22) %out_qubits_84 : !quantum.bit
    %23 = arith.extf %extracted_73 : f32 to f64
    %out_qubits_86 = quantum.custom "RZ"(%23) %out_qubits_85 : !quantum.bit
    %extracted_slice_87 = tensor.extract_slice %arg0[0, 3, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_88 = tensor.collapse_shape %extracted_slice_87 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_89 = tensor.extract %collapsed_88[] : tensor<f32>
    %extracted_slice_90 = tensor.extract_slice %arg0[0, 3, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_91 = tensor.collapse_shape %extracted_slice_90 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_92 = tensor.extract %collapsed_91[] : tensor<f32>
    %extracted_slice_93 = tensor.extract_slice %arg0[0, 3, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_94 = tensor.collapse_shape %extracted_slice_93 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_95 = tensor.extract %collapsed_94[] : tensor<f32>
    %extracted_slice_96 = tensor.extract_slice %2[3] [1] [1] : tensor<8xf32> to tensor<1xf32>
    %collapsed_97 = tensor.collapse_shape %extracted_slice_96 [] : tensor<1xf32> into tensor<f32>
    %extracted_98 = tensor.extract %collapsed_97[] : tensor<f32>
    %24 = quantum.extract %3[ 3] : !quantum.reg -> !quantum.bit
    %25 = arith.extf %extracted_98 : f32 to f64
    %out_qubits_99 = quantum.custom "RY"(%25) %24 : !quantum.bit
    %26 = arith.extf %extracted_95 : f32 to f64
    %out_qubits_100 = quantum.custom "RZ"(%26) %out_qubits_99 : !quantum.bit
    %27 = arith.extf %extracted_92 : f32 to f64
    %out_qubits_101 = quantum.custom "RY"(%27) %out_qubits_100 : !quantum.bit
    %28 = arith.extf %extracted_89 : f32 to f64
    %out_qubits_102 = quantum.custom "RZ"(%28) %out_qubits_101 : !quantum.bit
    %extracted_slice_103 = tensor.extract_slice %arg0[0, 2, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_104 = tensor.collapse_shape %extracted_slice_103 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_105 = tensor.extract %collapsed_104[] : tensor<f32>
    %extracted_slice_106 = tensor.extract_slice %arg0[0, 2, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_107 = tensor.collapse_shape %extracted_slice_106 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_108 = tensor.extract %collapsed_107[] : tensor<f32>
    %extracted_slice_109 = tensor.extract_slice %arg0[0, 2, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_110 = tensor.collapse_shape %extracted_slice_109 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_111 = tensor.extract %collapsed_110[] : tensor<f32>
    %extracted_slice_112 = tensor.extract_slice %2[2] [1] [1] : tensor<8xf32> to tensor<1xf32>
    %collapsed_113 = tensor.collapse_shape %extracted_slice_112 [] : tensor<1xf32> into tensor<f32>
    %extracted_114 = tensor.extract %collapsed_113[] : tensor<f32>
    %29 = quantum.extract %3[ 2] : !quantum.reg -> !quantum.bit
    %30 = arith.extf %extracted_114 : f32 to f64
    %out_qubits_115 = quantum.custom "RY"(%30) %29 : !quantum.bit
    %31 = arith.extf %extracted_111 : f32 to f64
    %out_qubits_116 = quantum.custom "RZ"(%31) %out_qubits_115 : !quantum.bit
    %32 = arith.extf %extracted_108 : f32 to f64
    %out_qubits_117 = quantum.custom "RY"(%32) %out_qubits_116 : !quantum.bit
    %33 = arith.extf %extracted_105 : f32 to f64
    %out_qubits_118 = quantum.custom "RZ"(%33) %out_qubits_117 : !quantum.bit
    %extracted_slice_119 = tensor.extract_slice %arg0[0, 0, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_120 = tensor.collapse_shape %extracted_slice_119 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_121 = tensor.extract %collapsed_120[] : tensor<f32>
    %extracted_slice_122 = tensor.extract_slice %arg0[0, 0, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_123 = tensor.collapse_shape %extracted_slice_122 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_124 = tensor.extract %collapsed_123[] : tensor<f32>
    %extracted_slice_125 = tensor.extract_slice %arg0[0, 0, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_126 = tensor.collapse_shape %extracted_slice_125 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_127 = tensor.extract %collapsed_126[] : tensor<f32>
    %extracted_slice_128 = tensor.extract_slice %2[0] [1] [1] : tensor<8xf32> to tensor<1xf32>
    %collapsed_129 = tensor.collapse_shape %extracted_slice_128 [] : tensor<1xf32> into tensor<f32>
    %extracted_130 = tensor.extract %collapsed_129[] : tensor<f32>
    %34 = quantum.extract %3[ 0] : !quantum.reg -> !quantum.bit
    %35 = arith.extf %extracted_130 : f32 to f64
    %out_qubits_131 = quantum.custom "RY"(%35) %34 : !quantum.bit
    %36 = arith.extf %extracted_127 : f32 to f64
    %out_qubits_132 = quantum.custom "RZ"(%36) %out_qubits_131 : !quantum.bit
    %37 = arith.extf %extracted_124 : f32 to f64
    %out_qubits_133 = quantum.custom "RY"(%37) %out_qubits_132 : !quantum.bit
    %38 = arith.extf %extracted_121 : f32 to f64
    %out_qubits_134 = quantum.custom "RZ"(%38) %out_qubits_133 : !quantum.bit
    %extracted_slice_135 = tensor.extract_slice %arg0[0, 1, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_136 = tensor.collapse_shape %extracted_slice_135 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_137 = tensor.extract %collapsed_136[] : tensor<f32>
    %extracted_slice_138 = tensor.extract_slice %arg0[0, 1, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_139 = tensor.collapse_shape %extracted_slice_138 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_140 = tensor.extract %collapsed_139[] : tensor<f32>
    %extracted_slice_141 = tensor.extract_slice %arg0[0, 1, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_142 = tensor.collapse_shape %extracted_slice_141 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_143 = tensor.extract %collapsed_142[] : tensor<f32>
    %extracted_slice_144 = tensor.extract_slice %2[1] [1] [1] : tensor<8xf32> to tensor<1xf32>
    %collapsed_145 = tensor.collapse_shape %extracted_slice_144 [] : tensor<1xf32> into tensor<f32>
    %extracted_146 = tensor.extract %collapsed_145[] : tensor<f32>
    %39 = quantum.extract %3[ 1] : !quantum.reg -> !quantum.bit
    %40 = arith.extf %extracted_146 : f32 to f64
    %out_qubits_147 = quantum.custom "RY"(%40) %39 : !quantum.bit
    %41 = arith.extf %extracted_143 : f32 to f64
    %out_qubits_148 = quantum.custom "RZ"(%41) %out_qubits_147 : !quantum.bit
    %42 = arith.extf %extracted_140 : f32 to f64
    %out_qubits_149 = quantum.custom "RY"(%42) %out_qubits_148 : !quantum.bit
    %43 = arith.extf %extracted_137 : f32 to f64
    %out_qubits_150 = quantum.custom "RZ"(%43) %out_qubits_149 : !quantum.bit
    %out_qubits_151:2 = quantum.custom "CNOT"() %out_qubits_134, %out_qubits_150 : !quantum.bit, !quantum.bit
    %out_qubits_152:2 = quantum.custom "CNOT"() %out_qubits_151#1, %out_qubits_118 : !quantum.bit, !quantum.bit
    %out_qubits_153:2 = quantum.custom "CNOT"() %out_qubits_152#1, %out_qubits_102 : !quantum.bit, !quantum.bit
    %out_qubits_154:2 = quantum.custom "CNOT"() %out_qubits_153#1, %out_qubits_86 : !quantum.bit, !quantum.bit
    %out_qubits_155:2 = quantum.custom "CNOT"() %out_qubits_154#1, %out_qubits_70 : !quantum.bit, !quantum.bit
    %out_qubits_156:2 = quantum.custom "CNOT"() %out_qubits_155#1, %out_qubits_54 : !quantum.bit, !quantum.bit
    %out_qubits_157:2 = quantum.custom "CNOT"() %out_qubits_156#1, %out_qubits_38 : !quantum.bit, !quantum.bit
    %44 = arith.extf %extracted_23 : f32 to f64
    %out_qubits_158 = quantum.custom "RZ"(%44) %out_qubits_157#0 : !quantum.bit
    %45 = arith.extf %extracted_20 : f32 to f64
    %out_qubits_159 = quantum.custom "RY"(%45) %out_qubits_158 : !quantum.bit
    %46 = arith.extf %extracted_17 : f32 to f64
    %out_qubits_160 = quantum.custom "RZ"(%46) %out_qubits_159 : !quantum.bit
    %extracted_slice_161 = tensor.extract_slice %arg0[1, 4, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_162 = tensor.collapse_shape %extracted_slice_161 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_163 = tensor.extract %collapsed_162[] : tensor<f32>
    %extracted_slice_164 = tensor.extract_slice %arg0[1, 4, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_165 = tensor.collapse_shape %extracted_slice_164 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_166 = tensor.extract %collapsed_165[] : tensor<f32>
    %extracted_slice_167 = tensor.extract_slice %arg0[1, 4, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_168 = tensor.collapse_shape %extracted_slice_167 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_169 = tensor.extract %collapsed_168[] : tensor<f32>
    %47 = arith.extf %extracted_169 : f32 to f64
    %out_qubits_170 = quantum.custom "RZ"(%47) %out_qubits_155#0 : !quantum.bit
    %48 = arith.extf %extracted_166 : f32 to f64
    %out_qubits_171 = quantum.custom "RY"(%48) %out_qubits_170 : !quantum.bit
    %49 = arith.extf %extracted_163 : f32 to f64
    %out_qubits_172 = quantum.custom "RZ"(%49) %out_qubits_171 : !quantum.bit
    %extracted_slice_173 = tensor.extract_slice %arg0[1, 0, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_174 = tensor.collapse_shape %extracted_slice_173 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_175 = tensor.extract %collapsed_174[] : tensor<f32>
    %extracted_slice_176 = tensor.extract_slice %arg0[1, 0, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_177 = tensor.collapse_shape %extracted_slice_176 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_178 = tensor.extract %collapsed_177[] : tensor<f32>
    %extracted_slice_179 = tensor.extract_slice %arg0[1, 0, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_180 = tensor.collapse_shape %extracted_slice_179 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_181 = tensor.extract %collapsed_180[] : tensor<f32>
    %out_qubits_182:2 = quantum.custom "CNOT"() %out_qubits_157#1, %out_qubits_151#0 : !quantum.bit, !quantum.bit
    %50 = arith.extf %extracted_181 : f32 to f64
    %out_qubits_183 = quantum.custom "RZ"(%50) %out_qubits_182#1 : !quantum.bit
    %51 = arith.extf %extracted_178 : f32 to f64
    %out_qubits_184 = quantum.custom "RY"(%51) %out_qubits_183 : !quantum.bit
    %52 = arith.extf %extracted_175 : f32 to f64
    %out_qubits_185 = quantum.custom "RZ"(%52) %out_qubits_184 : !quantum.bit
    %extracted_slice_186 = tensor.extract_slice %arg0[1, 2, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_187 = tensor.collapse_shape %extracted_slice_186 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_188 = tensor.extract %collapsed_187[] : tensor<f32>
    %extracted_slice_189 = tensor.extract_slice %arg0[1, 2, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_190 = tensor.collapse_shape %extracted_slice_189 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_191 = tensor.extract %collapsed_190[] : tensor<f32>
    %extracted_slice_192 = tensor.extract_slice %arg0[1, 2, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_193 = tensor.collapse_shape %extracted_slice_192 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_194 = tensor.extract %collapsed_193[] : tensor<f32>
    %53 = arith.extf %extracted_194 : f32 to f64
    %out_qubits_195 = quantum.custom "RZ"(%53) %out_qubits_153#0 : !quantum.bit
    %54 = arith.extf %extracted_191 : f32 to f64
    %out_qubits_196 = quantum.custom "RY"(%54) %out_qubits_195 : !quantum.bit
    %55 = arith.extf %extracted_188 : f32 to f64
    %out_qubits_197 = quantum.custom "RZ"(%55) %out_qubits_196 : !quantum.bit
    %out_qubits_198:2 = quantum.custom "CNOT"() %out_qubits_185, %out_qubits_197 : !quantum.bit, !quantum.bit
    %out_qubits_199:2 = quantum.custom "CNOT"() %out_qubits_198#1, %out_qubits_172 : !quantum.bit, !quantum.bit
    %out_qubits_200:2 = quantum.custom "CNOT"() %out_qubits_199#1, %out_qubits_160 : !quantum.bit, !quantum.bit
    %out_qubits_201:2 = quantum.custom "CNOT"() %out_qubits_200#1, %out_qubits_198#0 : !quantum.bit, !quantum.bit
    %56 = arith.extf %extracted_14 : f32 to f64
    %out_qubits_202 = quantum.custom "RZ"(%56) %out_qubits_201#1 : !quantum.bit
    %57 = arith.extf %extracted_11 : f32 to f64
    %out_qubits_203 = quantum.custom "RY"(%57) %out_qubits_202 : !quantum.bit
    %58 = arith.extf %extracted_8 : f32 to f64
    %out_qubits_204 = quantum.custom "RZ"(%58) %out_qubits_203 : !quantum.bit
    %extracted_slice_205 = tensor.extract_slice %arg0[2, 3, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_206 = tensor.collapse_shape %extracted_slice_205 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_207 = tensor.extract %collapsed_206[] : tensor<f32>
    %extracted_slice_208 = tensor.extract_slice %arg0[2, 3, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_209 = tensor.collapse_shape %extracted_slice_208 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_210 = tensor.extract %collapsed_209[] : tensor<f32>
    %extracted_slice_211 = tensor.extract_slice %arg0[2, 3, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_212 = tensor.collapse_shape %extracted_slice_211 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_213 = tensor.extract %collapsed_212[] : tensor<f32>
    %extracted_slice_214 = tensor.extract_slice %arg0[1, 5, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_215 = tensor.collapse_shape %extracted_slice_214 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_216 = tensor.extract %collapsed_215[] : tensor<f32>
    %extracted_slice_217 = tensor.extract_slice %arg0[1, 5, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_218 = tensor.collapse_shape %extracted_slice_217 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_219 = tensor.extract %collapsed_218[] : tensor<f32>
    %extracted_slice_220 = tensor.extract_slice %arg0[1, 5, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_221 = tensor.collapse_shape %extracted_slice_220 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_222 = tensor.extract %collapsed_221[] : tensor<f32>
    %59 = arith.extf %extracted_222 : f32 to f64
    %out_qubits_223 = quantum.custom "RZ"(%59) %out_qubits_156#0 : !quantum.bit
    %60 = arith.extf %extracted_219 : f32 to f64
    %out_qubits_224 = quantum.custom "RY"(%60) %out_qubits_223 : !quantum.bit
    %61 = arith.extf %extracted_216 : f32 to f64
    %out_qubits_225 = quantum.custom "RZ"(%61) %out_qubits_224 : !quantum.bit
    %extracted_slice_226 = tensor.extract_slice %arg0[1, 1, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_227 = tensor.collapse_shape %extracted_slice_226 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_228 = tensor.extract %collapsed_227[] : tensor<f32>
    %extracted_slice_229 = tensor.extract_slice %arg0[1, 1, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_230 = tensor.collapse_shape %extracted_slice_229 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_231 = tensor.extract %collapsed_230[] : tensor<f32>
    %extracted_slice_232 = tensor.extract_slice %arg0[1, 1, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_233 = tensor.collapse_shape %extracted_slice_232 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_234 = tensor.extract %collapsed_233[] : tensor<f32>
    %62 = arith.extf %extracted_234 : f32 to f64
    %out_qubits_235 = quantum.custom "RZ"(%62) %out_qubits_152#0 : !quantum.bit
    %63 = arith.extf %extracted_231 : f32 to f64
    %out_qubits_236 = quantum.custom "RY"(%63) %out_qubits_235 : !quantum.bit
    %64 = arith.extf %extracted_228 : f32 to f64
    %out_qubits_237 = quantum.custom "RZ"(%64) %out_qubits_236 : !quantum.bit
    %extracted_slice_238 = tensor.extract_slice %arg0[1, 3, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_239 = tensor.collapse_shape %extracted_slice_238 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_240 = tensor.extract %collapsed_239[] : tensor<f32>
    %extracted_slice_241 = tensor.extract_slice %arg0[1, 3, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_242 = tensor.collapse_shape %extracted_slice_241 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_243 = tensor.extract %collapsed_242[] : tensor<f32>
    %extracted_slice_244 = tensor.extract_slice %arg0[1, 3, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_245 = tensor.collapse_shape %extracted_slice_244 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_246 = tensor.extract %collapsed_245[] : tensor<f32>
    %65 = arith.extf %extracted_246 : f32 to f64
    %out_qubits_247 = quantum.custom "RZ"(%65) %out_qubits_154#0 : !quantum.bit
    %66 = arith.extf %extracted_243 : f32 to f64
    %out_qubits_248 = quantum.custom "RY"(%66) %out_qubits_247 : !quantum.bit
    %67 = arith.extf %extracted_240 : f32 to f64
    %out_qubits_249 = quantum.custom "RZ"(%67) %out_qubits_248 : !quantum.bit
    %out_qubits_250:2 = quantum.custom "CNOT"() %out_qubits_237, %out_qubits_249 : !quantum.bit, !quantum.bit
    %out_qubits_251:2 = quantum.custom "CNOT"() %out_qubits_250#1, %out_qubits_225 : !quantum.bit, !quantum.bit
    %68 = arith.extf %extracted_213 : f32 to f64
    %out_qubits_252 = quantum.custom "RZ"(%68) %out_qubits_251#0 : !quantum.bit
    %69 = arith.extf %extracted_210 : f32 to f64
    %out_qubits_253 = quantum.custom "RY"(%69) %out_qubits_252 : !quantum.bit
    %70 = arith.extf %extracted_207 : f32 to f64
    %out_qubits_254 = quantum.custom "RZ"(%70) %out_qubits_253 : !quantum.bit
    %out_qubits_255:2 = quantum.custom "CNOT"() %out_qubits_204, %out_qubits_254 : !quantum.bit, !quantum.bit
    %extracted_slice_256 = tensor.extract_slice %arg0[2, 2, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_257 = tensor.collapse_shape %extracted_slice_256 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_258 = tensor.extract %collapsed_257[] : tensor<f32>
    %extracted_slice_259 = tensor.extract_slice %arg0[2, 2, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_260 = tensor.collapse_shape %extracted_slice_259 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_261 = tensor.extract %collapsed_260[] : tensor<f32>
    %extracted_slice_262 = tensor.extract_slice %arg0[2, 2, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_263 = tensor.collapse_shape %extracted_slice_262 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_264 = tensor.extract %collapsed_263[] : tensor<f32>
    %71 = arith.extf %extracted_264 : f32 to f64
    %out_qubits_265 = quantum.custom "RZ"(%71) %out_qubits_199#0 : !quantum.bit
    %72 = arith.extf %extracted_261 : f32 to f64
    %out_qubits_266 = quantum.custom "RY"(%72) %out_qubits_265 : !quantum.bit
    %73 = arith.extf %extracted_258 : f32 to f64
    %out_qubits_267 = quantum.custom "RZ"(%73) %out_qubits_266 : !quantum.bit
    %extracted_slice_268 = tensor.extract_slice %arg0[2, 5, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_269 = tensor.collapse_shape %extracted_slice_268 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_270 = tensor.extract %collapsed_269[] : tensor<f32>
    %extracted_slice_271 = tensor.extract_slice %arg0[2, 5, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_272 = tensor.collapse_shape %extracted_slice_271 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_273 = tensor.extract %collapsed_272[] : tensor<f32>
    %extracted_slice_274 = tensor.extract_slice %arg0[2, 5, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_275 = tensor.collapse_shape %extracted_slice_274 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_276 = tensor.extract %collapsed_275[] : tensor<f32>
    %extracted_slice_277 = tensor.extract_slice %arg0[1, 7, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_278 = tensor.collapse_shape %extracted_slice_277 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_279 = tensor.extract %collapsed_278[] : tensor<f32>
    %extracted_slice_280 = tensor.extract_slice %arg0[1, 7, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_281 = tensor.collapse_shape %extracted_slice_280 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_282 = tensor.extract %collapsed_281[] : tensor<f32>
    %extracted_slice_283 = tensor.extract_slice %arg0[1, 7, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_284 = tensor.collapse_shape %extracted_slice_283 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_285 = tensor.extract %collapsed_284[] : tensor<f32>
    %74 = arith.extf %extracted_285 : f32 to f64
    %out_qubits_286 = quantum.custom "RZ"(%74) %out_qubits_182#0 : !quantum.bit
    %75 = arith.extf %extracted_282 : f32 to f64
    %out_qubits_287 = quantum.custom "RY"(%75) %out_qubits_286 : !quantum.bit
    %76 = arith.extf %extracted_279 : f32 to f64
    %out_qubits_288 = quantum.custom "RZ"(%76) %out_qubits_287 : !quantum.bit
    %out_qubits_289:2 = quantum.custom "CNOT"() %out_qubits_251#1, %out_qubits_288 : !quantum.bit, !quantum.bit
    %77 = arith.extf %extracted_276 : f32 to f64
    %out_qubits_290 = quantum.custom "RZ"(%77) %out_qubits_289#0 : !quantum.bit
    %78 = arith.extf %extracted_273 : f32 to f64
    %out_qubits_291 = quantum.custom "RY"(%78) %out_qubits_290 : !quantum.bit
    %79 = arith.extf %extracted_270 : f32 to f64
    %out_qubits_292 = quantum.custom "RZ"(%79) %out_qubits_291 : !quantum.bit
    %out_qubits_293:2 = quantum.custom "CNOT"() %out_qubits_267, %out_qubits_292 : !quantum.bit, !quantum.bit
    %out_qubits_294:2 = quantum.custom "CNOT"() %out_qubits_293#1, %out_qubits_255#0 : !quantum.bit, !quantum.bit
    %80 = arith.extf %extracted_5 : f32 to f64
    %out_qubits_295 = quantum.custom "RZ"(%80) %out_qubits_294#1 : !quantum.bit
    %81 = arith.extf %extracted_2 : f32 to f64
    %out_qubits_296 = quantum.custom "RY"(%81) %out_qubits_295 : !quantum.bit
    %82 = arith.extf %extracted : f32 to f64
    %out_qubits_297 = quantum.custom "RZ"(%82) %out_qubits_296 : !quantum.bit
    %extracted_slice_298 = tensor.extract_slice %arg0[3, 4, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_299 = tensor.collapse_shape %extracted_slice_298 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_300 = tensor.extract %collapsed_299[] : tensor<f32>
    %extracted_slice_301 = tensor.extract_slice %arg0[3, 4, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_302 = tensor.collapse_shape %extracted_slice_301 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_303 = tensor.extract %collapsed_302[] : tensor<f32>
    %extracted_slice_304 = tensor.extract_slice %arg0[3, 4, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_305 = tensor.collapse_shape %extracted_slice_304 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_306 = tensor.extract %collapsed_305[] : tensor<f32>
    %extracted_slice_307 = tensor.extract_slice %arg0[2, 7, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_308 = tensor.collapse_shape %extracted_slice_307 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_309 = tensor.extract %collapsed_308[] : tensor<f32>
    %extracted_slice_310 = tensor.extract_slice %arg0[2, 7, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_311 = tensor.collapse_shape %extracted_slice_310 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_312 = tensor.extract %collapsed_311[] : tensor<f32>
    %extracted_slice_313 = tensor.extract_slice %arg0[2, 7, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_314 = tensor.collapse_shape %extracted_slice_313 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_315 = tensor.extract %collapsed_314[] : tensor<f32>
    %out_qubits_316:2 = quantum.custom "CNOT"() %out_qubits_289#1, %out_qubits_250#0 : !quantum.bit, !quantum.bit
    %83 = arith.extf %extracted_315 : f32 to f64
    %out_qubits_317 = quantum.custom "RZ"(%83) %out_qubits_316#0 : !quantum.bit
    %84 = arith.extf %extracted_312 : f32 to f64
    %out_qubits_318 = quantum.custom "RY"(%84) %out_qubits_317 : !quantum.bit
    %85 = arith.extf %extracted_309 : f32 to f64
    %out_qubits_319 = quantum.custom "RZ"(%85) %out_qubits_318 : !quantum.bit
    %extracted_slice_320 = tensor.extract_slice %arg0[2, 1, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_321 = tensor.collapse_shape %extracted_slice_320 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_322 = tensor.extract %collapsed_321[] : tensor<f32>
    %extracted_slice_323 = tensor.extract_slice %arg0[2, 1, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_324 = tensor.collapse_shape %extracted_slice_323 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_325 = tensor.extract %collapsed_324[] : tensor<f32>
    %extracted_slice_326 = tensor.extract_slice %arg0[2, 1, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_327 = tensor.collapse_shape %extracted_slice_326 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_328 = tensor.extract %collapsed_327[] : tensor<f32>
    %86 = arith.extf %extracted_328 : f32 to f64
    %out_qubits_329 = quantum.custom "RZ"(%86) %out_qubits_316#1 : !quantum.bit
    %87 = arith.extf %extracted_325 : f32 to f64
    %out_qubits_330 = quantum.custom "RY"(%87) %out_qubits_329 : !quantum.bit
    %88 = arith.extf %extracted_322 : f32 to f64
    %out_qubits_331 = quantum.custom "RZ"(%88) %out_qubits_330 : !quantum.bit
    %extracted_slice_332 = tensor.extract_slice %arg0[2, 4, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_333 = tensor.collapse_shape %extracted_slice_332 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_334 = tensor.extract %collapsed_333[] : tensor<f32>
    %extracted_slice_335 = tensor.extract_slice %arg0[2, 4, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_336 = tensor.collapse_shape %extracted_slice_335 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_337 = tensor.extract %collapsed_336[] : tensor<f32>
    %extracted_slice_338 = tensor.extract_slice %arg0[2, 4, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_339 = tensor.collapse_shape %extracted_slice_338 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_340 = tensor.extract %collapsed_339[] : tensor<f32>
    %89 = arith.extf %extracted_340 : f32 to f64
    %out_qubits_341 = quantum.custom "RZ"(%89) %out_qubits_200#0 : !quantum.bit
    %90 = arith.extf %extracted_337 : f32 to f64
    %out_qubits_342 = quantum.custom "RY"(%90) %out_qubits_341 : !quantum.bit
    %91 = arith.extf %extracted_334 : f32 to f64
    %out_qubits_343 = quantum.custom "RZ"(%91) %out_qubits_342 : !quantum.bit
    %out_qubits_344:2 = quantum.custom "CNOT"() %out_qubits_331, %out_qubits_343 : !quantum.bit, !quantum.bit
    %out_qubits_345:2 = quantum.custom "CNOT"() %out_qubits_344#1, %out_qubits_319 : !quantum.bit, !quantum.bit
    %92 = arith.extf %extracted_306 : f32 to f64
    %out_qubits_346 = quantum.custom "RZ"(%92) %out_qubits_345#0 : !quantum.bit
    %93 = arith.extf %extracted_303 : f32 to f64
    %out_qubits_347 = quantum.custom "RY"(%93) %out_qubits_346 : !quantum.bit
    %94 = arith.extf %extracted_300 : f32 to f64
    %out_qubits_348 = quantum.custom "RZ"(%94) %out_qubits_347 : !quantum.bit
    %out_qubits_349:2 = quantum.custom "CNOT"() %out_qubits_297, %out_qubits_348 : !quantum.bit, !quantum.bit
    %out_qubits_350:2 = quantum.custom "CNOT"() %out_qubits_349#1, %out_qubits_349#0 : !quantum.bit, !quantum.bit
    %extracted_slice_351 = tensor.extract_slice %arg0[3, 1, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_352 = tensor.collapse_shape %extracted_slice_351 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_353 = tensor.extract %collapsed_352[] : tensor<f32>
    %extracted_slice_354 = tensor.extract_slice %arg0[3, 1, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_355 = tensor.collapse_shape %extracted_slice_354 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_356 = tensor.extract %collapsed_355[] : tensor<f32>
    %extracted_slice_357 = tensor.extract_slice %arg0[3, 1, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_358 = tensor.collapse_shape %extracted_slice_357 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_359 = tensor.extract %collapsed_358[] : tensor<f32>
    %extracted_slice_360 = tensor.extract_slice %arg0[2, 6, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_361 = tensor.collapse_shape %extracted_slice_360 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_362 = tensor.extract %collapsed_361[] : tensor<f32>
    %extracted_slice_363 = tensor.extract_slice %arg0[2, 6, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_364 = tensor.collapse_shape %extracted_slice_363 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_365 = tensor.extract %collapsed_364[] : tensor<f32>
    %extracted_slice_366 = tensor.extract_slice %arg0[2, 6, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_367 = tensor.collapse_shape %extracted_slice_366 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_368 = tensor.extract %collapsed_367[] : tensor<f32>
    %95 = arith.extf %extracted_368 : f32 to f64
    %out_qubits_369 = quantum.custom "RZ"(%95) %out_qubits_201#0 : !quantum.bit
    %96 = arith.extf %extracted_365 : f32 to f64
    %out_qubits_370 = quantum.custom "RY"(%96) %out_qubits_369 : !quantum.bit
    %97 = arith.extf %extracted_362 : f32 to f64
    %out_qubits_371 = quantum.custom "RZ"(%97) %out_qubits_370 : !quantum.bit
    %out_qubits_372:2 = quantum.custom "CNOT"() %out_qubits_255#1, %out_qubits_371 : !quantum.bit, !quantum.bit
    %out_qubits_373:2 = quantum.custom "CNOT"() %out_qubits_372#1, %out_qubits_344#0 : !quantum.bit, !quantum.bit
    %98 = arith.extf %extracted_359 : f32 to f64
    %out_qubits_374 = quantum.custom "RZ"(%98) %out_qubits_373#1 : !quantum.bit
    %99 = arith.extf %extracted_356 : f32 to f64
    %out_qubits_375 = quantum.custom "RY"(%99) %out_qubits_374 : !quantum.bit
    %100 = arith.extf %extracted_353 : f32 to f64
    %out_qubits_376 = quantum.custom "RZ"(%100) %out_qubits_375 : !quantum.bit
    %extracted_slice_377 = tensor.extract_slice %arg0[3, 5, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_378 = tensor.collapse_shape %extracted_slice_377 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_379 = tensor.extract %collapsed_378[] : tensor<f32>
    %extracted_slice_380 = tensor.extract_slice %arg0[3, 5, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_381 = tensor.collapse_shape %extracted_slice_380 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_382 = tensor.extract %collapsed_381[] : tensor<f32>
    %extracted_slice_383 = tensor.extract_slice %arg0[3, 5, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_384 = tensor.collapse_shape %extracted_slice_383 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_385 = tensor.extract %collapsed_384[] : tensor<f32>
    %101 = arith.extf %extracted_385 : f32 to f64
    %out_qubits_386 = quantum.custom "RZ"(%101) %out_qubits_294#0 : !quantum.bit
    %102 = arith.extf %extracted_382 : f32 to f64
    %out_qubits_387 = quantum.custom "RY"(%102) %out_qubits_386 : !quantum.bit
    %103 = arith.extf %extracted_379 : f32 to f64
    %out_qubits_388 = quantum.custom "RZ"(%103) %out_qubits_387 : !quantum.bit
    %out_qubits_389:2 = quantum.custom "CNOT"() %out_qubits_376, %out_qubits_388 : !quantum.bit, !quantum.bit
    %out_qubits_390:2 = quantum.custom "CNOT"() %out_qubits_389#1, %out_qubits_389#0 : !quantum.bit, !quantum.bit
    %extracted_slice_391 = tensor.extract_slice %arg0[3, 2, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_392 = tensor.collapse_shape %extracted_slice_391 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_393 = tensor.extract %collapsed_392[] : tensor<f32>
    %extracted_slice_394 = tensor.extract_slice %arg0[3, 2, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_395 = tensor.collapse_shape %extracted_slice_394 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_396 = tensor.extract %collapsed_395[] : tensor<f32>
    %extracted_slice_397 = tensor.extract_slice %arg0[3, 2, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_398 = tensor.collapse_shape %extracted_slice_397 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_399 = tensor.extract %collapsed_398[] : tensor<f32>
    %out_qubits_400:2 = quantum.custom "CNOT"() %out_qubits_345#1, %out_qubits_293#0 : !quantum.bit, !quantum.bit
    %104 = arith.extf %extracted_399 : f32 to f64
    %out_qubits_401 = quantum.custom "RZ"(%104) %out_qubits_400#1 : !quantum.bit
    %105 = arith.extf %extracted_396 : f32 to f64
    %out_qubits_402 = quantum.custom "RY"(%105) %out_qubits_401 : !quantum.bit
    %106 = arith.extf %extracted_393 : f32 to f64
    %out_qubits_403 = quantum.custom "RZ"(%106) %out_qubits_402 : !quantum.bit
    %extracted_slice_404 = tensor.extract_slice %arg0[3, 6, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_405 = tensor.collapse_shape %extracted_slice_404 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_406 = tensor.extract %collapsed_405[] : tensor<f32>
    %extracted_slice_407 = tensor.extract_slice %arg0[3, 6, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_408 = tensor.collapse_shape %extracted_slice_407 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_409 = tensor.extract %collapsed_408[] : tensor<f32>
    %extracted_slice_410 = tensor.extract_slice %arg0[3, 6, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_411 = tensor.collapse_shape %extracted_slice_410 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_412 = tensor.extract %collapsed_411[] : tensor<f32>
    %107 = arith.extf %extracted_412 : f32 to f64
    %out_qubits_413 = quantum.custom "RZ"(%107) %out_qubits_373#0 : !quantum.bit
    %108 = arith.extf %extracted_409 : f32 to f64
    %out_qubits_414 = quantum.custom "RY"(%108) %out_qubits_413 : !quantum.bit
    %109 = arith.extf %extracted_406 : f32 to f64
    %out_qubits_415 = quantum.custom "RZ"(%109) %out_qubits_414 : !quantum.bit
    %out_qubits_416:2 = quantum.custom "CNOT"() %out_qubits_403, %out_qubits_415 : !quantum.bit, !quantum.bit
    %out_qubits_417:2 = quantum.custom "CNOT"() %out_qubits_416#1, %out_qubits_416#0 : !quantum.bit, !quantum.bit
    %extracted_slice_418 = tensor.extract_slice %arg0[3, 3, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_419 = tensor.collapse_shape %extracted_slice_418 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_420 = tensor.extract %collapsed_419[] : tensor<f32>
    %extracted_slice_421 = tensor.extract_slice %arg0[3, 3, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_422 = tensor.collapse_shape %extracted_slice_421 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_423 = tensor.extract %collapsed_422[] : tensor<f32>
    %extracted_slice_424 = tensor.extract_slice %arg0[3, 3, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_425 = tensor.collapse_shape %extracted_slice_424 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_426 = tensor.extract %collapsed_425[] : tensor<f32>
    %110 = arith.extf %extracted_426 : f32 to f64
    %out_qubits_427 = quantum.custom "RZ"(%110) %out_qubits_372#0 : !quantum.bit
    %111 = arith.extf %extracted_423 : f32 to f64
    %out_qubits_428 = quantum.custom "RY"(%111) %out_qubits_427 : !quantum.bit
    %112 = arith.extf %extracted_420 : f32 to f64
    %out_qubits_429 = quantum.custom "RZ"(%112) %out_qubits_428 : !quantum.bit
    %extracted_slice_430 = tensor.extract_slice %arg0[3, 7, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_431 = tensor.collapse_shape %extracted_slice_430 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_432 = tensor.extract %collapsed_431[] : tensor<f32>
    %extracted_slice_433 = tensor.extract_slice %arg0[3, 7, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_434 = tensor.collapse_shape %extracted_slice_433 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_435 = tensor.extract %collapsed_434[] : tensor<f32>
    %extracted_slice_436 = tensor.extract_slice %arg0[3, 7, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_437 = tensor.collapse_shape %extracted_slice_436 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_438 = tensor.extract %collapsed_437[] : tensor<f32>
    %113 = arith.extf %extracted_438 : f32 to f64
    %out_qubits_439 = quantum.custom "RZ"(%113) %out_qubits_400#0 : !quantum.bit
    %114 = arith.extf %extracted_435 : f32 to f64
    %out_qubits_440 = quantum.custom "RY"(%114) %out_qubits_439 : !quantum.bit
    %115 = arith.extf %extracted_432 : f32 to f64
    %out_qubits_441 = quantum.custom "RZ"(%115) %out_qubits_440 : !quantum.bit
    %out_qubits_442:2 = quantum.custom "CNOT"() %out_qubits_429, %out_qubits_441 : !quantum.bit, !quantum.bit
    %out_qubits_443:2 = quantum.custom "CNOT"() %out_qubits_442#1, %out_qubits_442#0 : !quantum.bit, !quantum.bit
    %116 = quantum.namedobs %out_qubits_350#1[ PauliZ] : !quantum.obs
    %117 = quantum.expval %116 : f64
    %from_elements = tensor.from_elements %117 : tensor<f64>
    %118 = quantum.insert %3[ 0], %out_qubits_350#1 : !quantum.reg, !quantum.bit
    %119 = quantum.insert %118[ 1], %out_qubits_390#1 : !quantum.reg, !quantum.bit
    %120 = quantum.insert %119[ 2], %out_qubits_417#1 : !quantum.reg, !quantum.bit
    %121 = quantum.insert %120[ 3], %out_qubits_443#1 : !quantum.reg, !quantum.bit
    %122 = quantum.insert %121[ 4], %out_qubits_350#0 : !quantum.reg, !quantum.bit
    %123 = quantum.insert %122[ 5], %out_qubits_390#0 : !quantum.reg, !quantum.bit
    %124 = quantum.insert %123[ 6], %out_qubits_417#0 : !quantum.reg, !quantum.bit
    %125 = quantum.insert %124[ 7], %out_qubits_443#0 : !quantum.reg, !quantum.bit
    quantum.dealloc %125 : !quantum.reg
    quantum.device_release
    return %from_elements : tensor<f64>
  }
  func.func private @qnode_forward_0.adjoint(%arg0: tensor<4x8x3xf32>, %arg1: tensor<8xf32>, %arg2: index) -> tensor<?xf64> {
    %0 = gradient.adjoint @qnode_forward_0.nodealloc(%arg0, %arg1) size(%arg2) : (tensor<4x8x3xf32>, tensor<8xf32>) -> tensor<?xf64>
    return %0 : tensor<?xf64>
  }
  func.func private @qnode_forward_0.nodealloc(%arg0: tensor<4x8x3xf32>, %arg1: tensor<8xf32>) -> (!quantum.reg, f64) {
    %cst = arith.constant dense<3.14159274> : tensor<f32>
    %c0_i64 = arith.constant 0 : i64
    %extracted_slice = tensor.extract_slice %arg0[3, 0, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed = tensor.collapse_shape %extracted_slice [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted = tensor.extract %collapsed[] : tensor<f32>
    %extracted_slice_0 = tensor.extract_slice %arg0[3, 0, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_1 = tensor.collapse_shape %extracted_slice_0 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_2 = tensor.extract %collapsed_1[] : tensor<f32>
    %extracted_slice_3 = tensor.extract_slice %arg0[3, 0, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_4 = tensor.collapse_shape %extracted_slice_3 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_5 = tensor.extract %collapsed_4[] : tensor<f32>
    %extracted_slice_6 = tensor.extract_slice %arg0[2, 0, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_7 = tensor.collapse_shape %extracted_slice_6 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_8 = tensor.extract %collapsed_7[] : tensor<f32>
    %extracted_slice_9 = tensor.extract_slice %arg0[2, 0, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_10 = tensor.collapse_shape %extracted_slice_9 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_11 = tensor.extract %collapsed_10[] : tensor<f32>
    %extracted_slice_12 = tensor.extract_slice %arg0[2, 0, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_13 = tensor.collapse_shape %extracted_slice_12 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_14 = tensor.extract %collapsed_13[] : tensor<f32>
    %extracted_slice_15 = tensor.extract_slice %arg0[1, 6, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_16 = tensor.collapse_shape %extracted_slice_15 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_17 = tensor.extract %collapsed_16[] : tensor<f32>
    %extracted_slice_18 = tensor.extract_slice %arg0[1, 6, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_19 = tensor.collapse_shape %extracted_slice_18 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_20 = tensor.extract %collapsed_19[] : tensor<f32>
    %extracted_slice_21 = tensor.extract_slice %arg0[1, 6, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_22 = tensor.collapse_shape %extracted_slice_21 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_23 = tensor.extract %collapsed_22[] : tensor<f32>
    %extracted_slice_24 = tensor.extract_slice %arg0[0, 7, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_25 = tensor.collapse_shape %extracted_slice_24 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_26 = tensor.extract %collapsed_25[] : tensor<f32>
    %extracted_slice_27 = tensor.extract_slice %arg0[0, 7, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_28 = tensor.collapse_shape %extracted_slice_27 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_29 = tensor.extract %collapsed_28[] : tensor<f32>
    %extracted_slice_30 = tensor.extract_slice %arg0[0, 7, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_31 = tensor.collapse_shape %extracted_slice_30 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_32 = tensor.extract %collapsed_31[] : tensor<f32>
    %0 = tensor.empty() : tensor<8xf32>
    %1 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%cst : tensor<f32>) outs(%0 : tensor<8xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<8xf32>
    %2 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1, %arg1 : tensor<8xf32>, tensor<8xf32>) outs(%0 : tensor<8xf32>) {
    ^bb0(%in: f32, %in_444: f32, %out: f32):
      %126 = arith.mulf %in, %in_444 : f32
      linalg.yield %126 : f32
    } -> tensor<8xf32>
    %extracted_slice_33 = tensor.extract_slice %2[7] [1] [1] : tensor<8xf32> to tensor<1xf32>
    %collapsed_34 = tensor.collapse_shape %extracted_slice_33 [] : tensor<1xf32> into tensor<f32>
    %extracted_35 = tensor.extract %collapsed_34[] : tensor<f32>
    quantum.device shots(%c0_i64) ["/home/pichau/QAgents/.venv/lib/python3.12/site-packages/pennylane_lightning/liblightning_gpu_catalyst.so", "LightningGPUSimulator", "{}"]
    %3 = quantum.alloc( 8) : !quantum.reg
    %4 = quantum.extract %3[ 7] : !quantum.reg -> !quantum.bit
    %5 = arith.extf %extracted_35 : f32 to f64
    %out_qubits = quantum.custom "RY"(%5) %4 : !quantum.bit
    %6 = arith.extf %extracted_32 : f32 to f64
    %out_qubits_36 = quantum.custom "RZ"(%6) %out_qubits : !quantum.bit
    %7 = arith.extf %extracted_29 : f32 to f64
    %out_qubits_37 = quantum.custom "RY"(%7) %out_qubits_36 : !quantum.bit
    %8 = arith.extf %extracted_26 : f32 to f64
    %out_qubits_38 = quantum.custom "RZ"(%8) %out_qubits_37 : !quantum.bit
    %extracted_slice_39 = tensor.extract_slice %arg0[0, 6, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_40 = tensor.collapse_shape %extracted_slice_39 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_41 = tensor.extract %collapsed_40[] : tensor<f32>
    %extracted_slice_42 = tensor.extract_slice %arg0[0, 6, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_43 = tensor.collapse_shape %extracted_slice_42 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_44 = tensor.extract %collapsed_43[] : tensor<f32>
    %extracted_slice_45 = tensor.extract_slice %arg0[0, 6, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_46 = tensor.collapse_shape %extracted_slice_45 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_47 = tensor.extract %collapsed_46[] : tensor<f32>
    %extracted_slice_48 = tensor.extract_slice %2[6] [1] [1] : tensor<8xf32> to tensor<1xf32>
    %collapsed_49 = tensor.collapse_shape %extracted_slice_48 [] : tensor<1xf32> into tensor<f32>
    %extracted_50 = tensor.extract %collapsed_49[] : tensor<f32>
    %9 = quantum.extract %3[ 6] : !quantum.reg -> !quantum.bit
    %10 = arith.extf %extracted_50 : f32 to f64
    %out_qubits_51 = quantum.custom "RY"(%10) %9 : !quantum.bit
    %11 = arith.extf %extracted_47 : f32 to f64
    %out_qubits_52 = quantum.custom "RZ"(%11) %out_qubits_51 : !quantum.bit
    %12 = arith.extf %extracted_44 : f32 to f64
    %out_qubits_53 = quantum.custom "RY"(%12) %out_qubits_52 : !quantum.bit
    %13 = arith.extf %extracted_41 : f32 to f64
    %out_qubits_54 = quantum.custom "RZ"(%13) %out_qubits_53 : !quantum.bit
    %extracted_slice_55 = tensor.extract_slice %arg0[0, 5, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_56 = tensor.collapse_shape %extracted_slice_55 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_57 = tensor.extract %collapsed_56[] : tensor<f32>
    %extracted_slice_58 = tensor.extract_slice %arg0[0, 5, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_59 = tensor.collapse_shape %extracted_slice_58 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_60 = tensor.extract %collapsed_59[] : tensor<f32>
    %extracted_slice_61 = tensor.extract_slice %arg0[0, 5, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_62 = tensor.collapse_shape %extracted_slice_61 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_63 = tensor.extract %collapsed_62[] : tensor<f32>
    %extracted_slice_64 = tensor.extract_slice %2[5] [1] [1] : tensor<8xf32> to tensor<1xf32>
    %collapsed_65 = tensor.collapse_shape %extracted_slice_64 [] : tensor<1xf32> into tensor<f32>
    %extracted_66 = tensor.extract %collapsed_65[] : tensor<f32>
    %14 = quantum.extract %3[ 5] : !quantum.reg -> !quantum.bit
    %15 = arith.extf %extracted_66 : f32 to f64
    %out_qubits_67 = quantum.custom "RY"(%15) %14 : !quantum.bit
    %16 = arith.extf %extracted_63 : f32 to f64
    %out_qubits_68 = quantum.custom "RZ"(%16) %out_qubits_67 : !quantum.bit
    %17 = arith.extf %extracted_60 : f32 to f64
    %out_qubits_69 = quantum.custom "RY"(%17) %out_qubits_68 : !quantum.bit
    %18 = arith.extf %extracted_57 : f32 to f64
    %out_qubits_70 = quantum.custom "RZ"(%18) %out_qubits_69 : !quantum.bit
    %extracted_slice_71 = tensor.extract_slice %arg0[0, 4, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_72 = tensor.collapse_shape %extracted_slice_71 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_73 = tensor.extract %collapsed_72[] : tensor<f32>
    %extracted_slice_74 = tensor.extract_slice %arg0[0, 4, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_75 = tensor.collapse_shape %extracted_slice_74 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_76 = tensor.extract %collapsed_75[] : tensor<f32>
    %extracted_slice_77 = tensor.extract_slice %arg0[0, 4, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_78 = tensor.collapse_shape %extracted_slice_77 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_79 = tensor.extract %collapsed_78[] : tensor<f32>
    %extracted_slice_80 = tensor.extract_slice %2[4] [1] [1] : tensor<8xf32> to tensor<1xf32>
    %collapsed_81 = tensor.collapse_shape %extracted_slice_80 [] : tensor<1xf32> into tensor<f32>
    %extracted_82 = tensor.extract %collapsed_81[] : tensor<f32>
    %19 = quantum.extract %3[ 4] : !quantum.reg -> !quantum.bit
    %20 = arith.extf %extracted_82 : f32 to f64
    %out_qubits_83 = quantum.custom "RY"(%20) %19 : !quantum.bit
    %21 = arith.extf %extracted_79 : f32 to f64
    %out_qubits_84 = quantum.custom "RZ"(%21) %out_qubits_83 : !quantum.bit
    %22 = arith.extf %extracted_76 : f32 to f64
    %out_qubits_85 = quantum.custom "RY"(%22) %out_qubits_84 : !quantum.bit
    %23 = arith.extf %extracted_73 : f32 to f64
    %out_qubits_86 = quantum.custom "RZ"(%23) %out_qubits_85 : !quantum.bit
    %extracted_slice_87 = tensor.extract_slice %arg0[0, 3, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_88 = tensor.collapse_shape %extracted_slice_87 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_89 = tensor.extract %collapsed_88[] : tensor<f32>
    %extracted_slice_90 = tensor.extract_slice %arg0[0, 3, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_91 = tensor.collapse_shape %extracted_slice_90 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_92 = tensor.extract %collapsed_91[] : tensor<f32>
    %extracted_slice_93 = tensor.extract_slice %arg0[0, 3, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_94 = tensor.collapse_shape %extracted_slice_93 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_95 = tensor.extract %collapsed_94[] : tensor<f32>
    %extracted_slice_96 = tensor.extract_slice %2[3] [1] [1] : tensor<8xf32> to tensor<1xf32>
    %collapsed_97 = tensor.collapse_shape %extracted_slice_96 [] : tensor<1xf32> into tensor<f32>
    %extracted_98 = tensor.extract %collapsed_97[] : tensor<f32>
    %24 = quantum.extract %3[ 3] : !quantum.reg -> !quantum.bit
    %25 = arith.extf %extracted_98 : f32 to f64
    %out_qubits_99 = quantum.custom "RY"(%25) %24 : !quantum.bit
    %26 = arith.extf %extracted_95 : f32 to f64
    %out_qubits_100 = quantum.custom "RZ"(%26) %out_qubits_99 : !quantum.bit
    %27 = arith.extf %extracted_92 : f32 to f64
    %out_qubits_101 = quantum.custom "RY"(%27) %out_qubits_100 : !quantum.bit
    %28 = arith.extf %extracted_89 : f32 to f64
    %out_qubits_102 = quantum.custom "RZ"(%28) %out_qubits_101 : !quantum.bit
    %extracted_slice_103 = tensor.extract_slice %arg0[0, 2, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_104 = tensor.collapse_shape %extracted_slice_103 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_105 = tensor.extract %collapsed_104[] : tensor<f32>
    %extracted_slice_106 = tensor.extract_slice %arg0[0, 2, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_107 = tensor.collapse_shape %extracted_slice_106 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_108 = tensor.extract %collapsed_107[] : tensor<f32>
    %extracted_slice_109 = tensor.extract_slice %arg0[0, 2, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_110 = tensor.collapse_shape %extracted_slice_109 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_111 = tensor.extract %collapsed_110[] : tensor<f32>
    %extracted_slice_112 = tensor.extract_slice %2[2] [1] [1] : tensor<8xf32> to tensor<1xf32>
    %collapsed_113 = tensor.collapse_shape %extracted_slice_112 [] : tensor<1xf32> into tensor<f32>
    %extracted_114 = tensor.extract %collapsed_113[] : tensor<f32>
    %29 = quantum.extract %3[ 2] : !quantum.reg -> !quantum.bit
    %30 = arith.extf %extracted_114 : f32 to f64
    %out_qubits_115 = quantum.custom "RY"(%30) %29 : !quantum.bit
    %31 = arith.extf %extracted_111 : f32 to f64
    %out_qubits_116 = quantum.custom "RZ"(%31) %out_qubits_115 : !quantum.bit
    %32 = arith.extf %extracted_108 : f32 to f64
    %out_qubits_117 = quantum.custom "RY"(%32) %out_qubits_116 : !quantum.bit
    %33 = arith.extf %extracted_105 : f32 to f64
    %out_qubits_118 = quantum.custom "RZ"(%33) %out_qubits_117 : !quantum.bit
    %extracted_slice_119 = tensor.extract_slice %arg0[0, 0, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_120 = tensor.collapse_shape %extracted_slice_119 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_121 = tensor.extract %collapsed_120[] : tensor<f32>
    %extracted_slice_122 = tensor.extract_slice %arg0[0, 0, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_123 = tensor.collapse_shape %extracted_slice_122 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_124 = tensor.extract %collapsed_123[] : tensor<f32>
    %extracted_slice_125 = tensor.extract_slice %arg0[0, 0, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_126 = tensor.collapse_shape %extracted_slice_125 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_127 = tensor.extract %collapsed_126[] : tensor<f32>
    %extracted_slice_128 = tensor.extract_slice %2[0] [1] [1] : tensor<8xf32> to tensor<1xf32>
    %collapsed_129 = tensor.collapse_shape %extracted_slice_128 [] : tensor<1xf32> into tensor<f32>
    %extracted_130 = tensor.extract %collapsed_129[] : tensor<f32>
    %34 = quantum.extract %3[ 0] : !quantum.reg -> !quantum.bit
    %35 = arith.extf %extracted_130 : f32 to f64
    %out_qubits_131 = quantum.custom "RY"(%35) %34 : !quantum.bit
    %36 = arith.extf %extracted_127 : f32 to f64
    %out_qubits_132 = quantum.custom "RZ"(%36) %out_qubits_131 : !quantum.bit
    %37 = arith.extf %extracted_124 : f32 to f64
    %out_qubits_133 = quantum.custom "RY"(%37) %out_qubits_132 : !quantum.bit
    %38 = arith.extf %extracted_121 : f32 to f64
    %out_qubits_134 = quantum.custom "RZ"(%38) %out_qubits_133 : !quantum.bit
    %extracted_slice_135 = tensor.extract_slice %arg0[0, 1, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_136 = tensor.collapse_shape %extracted_slice_135 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_137 = tensor.extract %collapsed_136[] : tensor<f32>
    %extracted_slice_138 = tensor.extract_slice %arg0[0, 1, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_139 = tensor.collapse_shape %extracted_slice_138 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_140 = tensor.extract %collapsed_139[] : tensor<f32>
    %extracted_slice_141 = tensor.extract_slice %arg0[0, 1, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_142 = tensor.collapse_shape %extracted_slice_141 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_143 = tensor.extract %collapsed_142[] : tensor<f32>
    %extracted_slice_144 = tensor.extract_slice %2[1] [1] [1] : tensor<8xf32> to tensor<1xf32>
    %collapsed_145 = tensor.collapse_shape %extracted_slice_144 [] : tensor<1xf32> into tensor<f32>
    %extracted_146 = tensor.extract %collapsed_145[] : tensor<f32>
    %39 = quantum.extract %3[ 1] : !quantum.reg -> !quantum.bit
    %40 = arith.extf %extracted_146 : f32 to f64
    %out_qubits_147 = quantum.custom "RY"(%40) %39 : !quantum.bit
    %41 = arith.extf %extracted_143 : f32 to f64
    %out_qubits_148 = quantum.custom "RZ"(%41) %out_qubits_147 : !quantum.bit
    %42 = arith.extf %extracted_140 : f32 to f64
    %out_qubits_149 = quantum.custom "RY"(%42) %out_qubits_148 : !quantum.bit
    %43 = arith.extf %extracted_137 : f32 to f64
    %out_qubits_150 = quantum.custom "RZ"(%43) %out_qubits_149 : !quantum.bit
    %out_qubits_151:2 = quantum.custom "CNOT"() %out_qubits_134, %out_qubits_150 : !quantum.bit, !quantum.bit
    %out_qubits_152:2 = quantum.custom "CNOT"() %out_qubits_151#1, %out_qubits_118 : !quantum.bit, !quantum.bit
    %out_qubits_153:2 = quantum.custom "CNOT"() %out_qubits_152#1, %out_qubits_102 : !quantum.bit, !quantum.bit
    %out_qubits_154:2 = quantum.custom "CNOT"() %out_qubits_153#1, %out_qubits_86 : !quantum.bit, !quantum.bit
    %out_qubits_155:2 = quantum.custom "CNOT"() %out_qubits_154#1, %out_qubits_70 : !quantum.bit, !quantum.bit
    %out_qubits_156:2 = quantum.custom "CNOT"() %out_qubits_155#1, %out_qubits_54 : !quantum.bit, !quantum.bit
    %out_qubits_157:2 = quantum.custom "CNOT"() %out_qubits_156#1, %out_qubits_38 : !quantum.bit, !quantum.bit
    %44 = arith.extf %extracted_23 : f32 to f64
    %out_qubits_158 = quantum.custom "RZ"(%44) %out_qubits_157#0 : !quantum.bit
    %45 = arith.extf %extracted_20 : f32 to f64
    %out_qubits_159 = quantum.custom "RY"(%45) %out_qubits_158 : !quantum.bit
    %46 = arith.extf %extracted_17 : f32 to f64
    %out_qubits_160 = quantum.custom "RZ"(%46) %out_qubits_159 : !quantum.bit
    %extracted_slice_161 = tensor.extract_slice %arg0[1, 4, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_162 = tensor.collapse_shape %extracted_slice_161 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_163 = tensor.extract %collapsed_162[] : tensor<f32>
    %extracted_slice_164 = tensor.extract_slice %arg0[1, 4, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_165 = tensor.collapse_shape %extracted_slice_164 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_166 = tensor.extract %collapsed_165[] : tensor<f32>
    %extracted_slice_167 = tensor.extract_slice %arg0[1, 4, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_168 = tensor.collapse_shape %extracted_slice_167 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_169 = tensor.extract %collapsed_168[] : tensor<f32>
    %47 = arith.extf %extracted_169 : f32 to f64
    %out_qubits_170 = quantum.custom "RZ"(%47) %out_qubits_155#0 : !quantum.bit
    %48 = arith.extf %extracted_166 : f32 to f64
    %out_qubits_171 = quantum.custom "RY"(%48) %out_qubits_170 : !quantum.bit
    %49 = arith.extf %extracted_163 : f32 to f64
    %out_qubits_172 = quantum.custom "RZ"(%49) %out_qubits_171 : !quantum.bit
    %extracted_slice_173 = tensor.extract_slice %arg0[1, 0, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_174 = tensor.collapse_shape %extracted_slice_173 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_175 = tensor.extract %collapsed_174[] : tensor<f32>
    %extracted_slice_176 = tensor.extract_slice %arg0[1, 0, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_177 = tensor.collapse_shape %extracted_slice_176 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_178 = tensor.extract %collapsed_177[] : tensor<f32>
    %extracted_slice_179 = tensor.extract_slice %arg0[1, 0, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_180 = tensor.collapse_shape %extracted_slice_179 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_181 = tensor.extract %collapsed_180[] : tensor<f32>
    %out_qubits_182:2 = quantum.custom "CNOT"() %out_qubits_157#1, %out_qubits_151#0 : !quantum.bit, !quantum.bit
    %50 = arith.extf %extracted_181 : f32 to f64
    %out_qubits_183 = quantum.custom "RZ"(%50) %out_qubits_182#1 : !quantum.bit
    %51 = arith.extf %extracted_178 : f32 to f64
    %out_qubits_184 = quantum.custom "RY"(%51) %out_qubits_183 : !quantum.bit
    %52 = arith.extf %extracted_175 : f32 to f64
    %out_qubits_185 = quantum.custom "RZ"(%52) %out_qubits_184 : !quantum.bit
    %extracted_slice_186 = tensor.extract_slice %arg0[1, 2, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_187 = tensor.collapse_shape %extracted_slice_186 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_188 = tensor.extract %collapsed_187[] : tensor<f32>
    %extracted_slice_189 = tensor.extract_slice %arg0[1, 2, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_190 = tensor.collapse_shape %extracted_slice_189 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_191 = tensor.extract %collapsed_190[] : tensor<f32>
    %extracted_slice_192 = tensor.extract_slice %arg0[1, 2, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_193 = tensor.collapse_shape %extracted_slice_192 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_194 = tensor.extract %collapsed_193[] : tensor<f32>
    %53 = arith.extf %extracted_194 : f32 to f64
    %out_qubits_195 = quantum.custom "RZ"(%53) %out_qubits_153#0 : !quantum.bit
    %54 = arith.extf %extracted_191 : f32 to f64
    %out_qubits_196 = quantum.custom "RY"(%54) %out_qubits_195 : !quantum.bit
    %55 = arith.extf %extracted_188 : f32 to f64
    %out_qubits_197 = quantum.custom "RZ"(%55) %out_qubits_196 : !quantum.bit
    %out_qubits_198:2 = quantum.custom "CNOT"() %out_qubits_185, %out_qubits_197 : !quantum.bit, !quantum.bit
    %out_qubits_199:2 = quantum.custom "CNOT"() %out_qubits_198#1, %out_qubits_172 : !quantum.bit, !quantum.bit
    %out_qubits_200:2 = quantum.custom "CNOT"() %out_qubits_199#1, %out_qubits_160 : !quantum.bit, !quantum.bit
    %out_qubits_201:2 = quantum.custom "CNOT"() %out_qubits_200#1, %out_qubits_198#0 : !quantum.bit, !quantum.bit
    %56 = arith.extf %extracted_14 : f32 to f64
    %out_qubits_202 = quantum.custom "RZ"(%56) %out_qubits_201#1 : !quantum.bit
    %57 = arith.extf %extracted_11 : f32 to f64
    %out_qubits_203 = quantum.custom "RY"(%57) %out_qubits_202 : !quantum.bit
    %58 = arith.extf %extracted_8 : f32 to f64
    %out_qubits_204 = quantum.custom "RZ"(%58) %out_qubits_203 : !quantum.bit
    %extracted_slice_205 = tensor.extract_slice %arg0[2, 3, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_206 = tensor.collapse_shape %extracted_slice_205 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_207 = tensor.extract %collapsed_206[] : tensor<f32>
    %extracted_slice_208 = tensor.extract_slice %arg0[2, 3, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_209 = tensor.collapse_shape %extracted_slice_208 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_210 = tensor.extract %collapsed_209[] : tensor<f32>
    %extracted_slice_211 = tensor.extract_slice %arg0[2, 3, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_212 = tensor.collapse_shape %extracted_slice_211 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_213 = tensor.extract %collapsed_212[] : tensor<f32>
    %extracted_slice_214 = tensor.extract_slice %arg0[1, 5, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_215 = tensor.collapse_shape %extracted_slice_214 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_216 = tensor.extract %collapsed_215[] : tensor<f32>
    %extracted_slice_217 = tensor.extract_slice %arg0[1, 5, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_218 = tensor.collapse_shape %extracted_slice_217 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_219 = tensor.extract %collapsed_218[] : tensor<f32>
    %extracted_slice_220 = tensor.extract_slice %arg0[1, 5, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_221 = tensor.collapse_shape %extracted_slice_220 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_222 = tensor.extract %collapsed_221[] : tensor<f32>
    %59 = arith.extf %extracted_222 : f32 to f64
    %out_qubits_223 = quantum.custom "RZ"(%59) %out_qubits_156#0 : !quantum.bit
    %60 = arith.extf %extracted_219 : f32 to f64
    %out_qubits_224 = quantum.custom "RY"(%60) %out_qubits_223 : !quantum.bit
    %61 = arith.extf %extracted_216 : f32 to f64
    %out_qubits_225 = quantum.custom "RZ"(%61) %out_qubits_224 : !quantum.bit
    %extracted_slice_226 = tensor.extract_slice %arg0[1, 1, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_227 = tensor.collapse_shape %extracted_slice_226 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_228 = tensor.extract %collapsed_227[] : tensor<f32>
    %extracted_slice_229 = tensor.extract_slice %arg0[1, 1, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_230 = tensor.collapse_shape %extracted_slice_229 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_231 = tensor.extract %collapsed_230[] : tensor<f32>
    %extracted_slice_232 = tensor.extract_slice %arg0[1, 1, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_233 = tensor.collapse_shape %extracted_slice_232 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_234 = tensor.extract %collapsed_233[] : tensor<f32>
    %62 = arith.extf %extracted_234 : f32 to f64
    %out_qubits_235 = quantum.custom "RZ"(%62) %out_qubits_152#0 : !quantum.bit
    %63 = arith.extf %extracted_231 : f32 to f64
    %out_qubits_236 = quantum.custom "RY"(%63) %out_qubits_235 : !quantum.bit
    %64 = arith.extf %extracted_228 : f32 to f64
    %out_qubits_237 = quantum.custom "RZ"(%64) %out_qubits_236 : !quantum.bit
    %extracted_slice_238 = tensor.extract_slice %arg0[1, 3, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_239 = tensor.collapse_shape %extracted_slice_238 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_240 = tensor.extract %collapsed_239[] : tensor<f32>
    %extracted_slice_241 = tensor.extract_slice %arg0[1, 3, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_242 = tensor.collapse_shape %extracted_slice_241 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_243 = tensor.extract %collapsed_242[] : tensor<f32>
    %extracted_slice_244 = tensor.extract_slice %arg0[1, 3, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_245 = tensor.collapse_shape %extracted_slice_244 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_246 = tensor.extract %collapsed_245[] : tensor<f32>
    %65 = arith.extf %extracted_246 : f32 to f64
    %out_qubits_247 = quantum.custom "RZ"(%65) %out_qubits_154#0 : !quantum.bit
    %66 = arith.extf %extracted_243 : f32 to f64
    %out_qubits_248 = quantum.custom "RY"(%66) %out_qubits_247 : !quantum.bit
    %67 = arith.extf %extracted_240 : f32 to f64
    %out_qubits_249 = quantum.custom "RZ"(%67) %out_qubits_248 : !quantum.bit
    %out_qubits_250:2 = quantum.custom "CNOT"() %out_qubits_237, %out_qubits_249 : !quantum.bit, !quantum.bit
    %out_qubits_251:2 = quantum.custom "CNOT"() %out_qubits_250#1, %out_qubits_225 : !quantum.bit, !quantum.bit
    %68 = arith.extf %extracted_213 : f32 to f64
    %out_qubits_252 = quantum.custom "RZ"(%68) %out_qubits_251#0 : !quantum.bit
    %69 = arith.extf %extracted_210 : f32 to f64
    %out_qubits_253 = quantum.custom "RY"(%69) %out_qubits_252 : !quantum.bit
    %70 = arith.extf %extracted_207 : f32 to f64
    %out_qubits_254 = quantum.custom "RZ"(%70) %out_qubits_253 : !quantum.bit
    %out_qubits_255:2 = quantum.custom "CNOT"() %out_qubits_204, %out_qubits_254 : !quantum.bit, !quantum.bit
    %extracted_slice_256 = tensor.extract_slice %arg0[2, 2, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_257 = tensor.collapse_shape %extracted_slice_256 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_258 = tensor.extract %collapsed_257[] : tensor<f32>
    %extracted_slice_259 = tensor.extract_slice %arg0[2, 2, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_260 = tensor.collapse_shape %extracted_slice_259 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_261 = tensor.extract %collapsed_260[] : tensor<f32>
    %extracted_slice_262 = tensor.extract_slice %arg0[2, 2, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_263 = tensor.collapse_shape %extracted_slice_262 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_264 = tensor.extract %collapsed_263[] : tensor<f32>
    %71 = arith.extf %extracted_264 : f32 to f64
    %out_qubits_265 = quantum.custom "RZ"(%71) %out_qubits_199#0 : !quantum.bit
    %72 = arith.extf %extracted_261 : f32 to f64
    %out_qubits_266 = quantum.custom "RY"(%72) %out_qubits_265 : !quantum.bit
    %73 = arith.extf %extracted_258 : f32 to f64
    %out_qubits_267 = quantum.custom "RZ"(%73) %out_qubits_266 : !quantum.bit
    %extracted_slice_268 = tensor.extract_slice %arg0[2, 5, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_269 = tensor.collapse_shape %extracted_slice_268 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_270 = tensor.extract %collapsed_269[] : tensor<f32>
    %extracted_slice_271 = tensor.extract_slice %arg0[2, 5, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_272 = tensor.collapse_shape %extracted_slice_271 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_273 = tensor.extract %collapsed_272[] : tensor<f32>
    %extracted_slice_274 = tensor.extract_slice %arg0[2, 5, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_275 = tensor.collapse_shape %extracted_slice_274 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_276 = tensor.extract %collapsed_275[] : tensor<f32>
    %extracted_slice_277 = tensor.extract_slice %arg0[1, 7, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_278 = tensor.collapse_shape %extracted_slice_277 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_279 = tensor.extract %collapsed_278[] : tensor<f32>
    %extracted_slice_280 = tensor.extract_slice %arg0[1, 7, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_281 = tensor.collapse_shape %extracted_slice_280 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_282 = tensor.extract %collapsed_281[] : tensor<f32>
    %extracted_slice_283 = tensor.extract_slice %arg0[1, 7, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_284 = tensor.collapse_shape %extracted_slice_283 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_285 = tensor.extract %collapsed_284[] : tensor<f32>
    %74 = arith.extf %extracted_285 : f32 to f64
    %out_qubits_286 = quantum.custom "RZ"(%74) %out_qubits_182#0 : !quantum.bit
    %75 = arith.extf %extracted_282 : f32 to f64
    %out_qubits_287 = quantum.custom "RY"(%75) %out_qubits_286 : !quantum.bit
    %76 = arith.extf %extracted_279 : f32 to f64
    %out_qubits_288 = quantum.custom "RZ"(%76) %out_qubits_287 : !quantum.bit
    %out_qubits_289:2 = quantum.custom "CNOT"() %out_qubits_251#1, %out_qubits_288 : !quantum.bit, !quantum.bit
    %77 = arith.extf %extracted_276 : f32 to f64
    %out_qubits_290 = quantum.custom "RZ"(%77) %out_qubits_289#0 : !quantum.bit
    %78 = arith.extf %extracted_273 : f32 to f64
    %out_qubits_291 = quantum.custom "RY"(%78) %out_qubits_290 : !quantum.bit
    %79 = arith.extf %extracted_270 : f32 to f64
    %out_qubits_292 = quantum.custom "RZ"(%79) %out_qubits_291 : !quantum.bit
    %out_qubits_293:2 = quantum.custom "CNOT"() %out_qubits_267, %out_qubits_292 : !quantum.bit, !quantum.bit
    %out_qubits_294:2 = quantum.custom "CNOT"() %out_qubits_293#1, %out_qubits_255#0 : !quantum.bit, !quantum.bit
    %80 = arith.extf %extracted_5 : f32 to f64
    %out_qubits_295 = quantum.custom "RZ"(%80) %out_qubits_294#1 : !quantum.bit
    %81 = arith.extf %extracted_2 : f32 to f64
    %out_qubits_296 = quantum.custom "RY"(%81) %out_qubits_295 : !quantum.bit
    %82 = arith.extf %extracted : f32 to f64
    %out_qubits_297 = quantum.custom "RZ"(%82) %out_qubits_296 : !quantum.bit
    %extracted_slice_298 = tensor.extract_slice %arg0[3, 4, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_299 = tensor.collapse_shape %extracted_slice_298 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_300 = tensor.extract %collapsed_299[] : tensor<f32>
    %extracted_slice_301 = tensor.extract_slice %arg0[3, 4, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_302 = tensor.collapse_shape %extracted_slice_301 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_303 = tensor.extract %collapsed_302[] : tensor<f32>
    %extracted_slice_304 = tensor.extract_slice %arg0[3, 4, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_305 = tensor.collapse_shape %extracted_slice_304 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_306 = tensor.extract %collapsed_305[] : tensor<f32>
    %extracted_slice_307 = tensor.extract_slice %arg0[2, 7, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_308 = tensor.collapse_shape %extracted_slice_307 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_309 = tensor.extract %collapsed_308[] : tensor<f32>
    %extracted_slice_310 = tensor.extract_slice %arg0[2, 7, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_311 = tensor.collapse_shape %extracted_slice_310 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_312 = tensor.extract %collapsed_311[] : tensor<f32>
    %extracted_slice_313 = tensor.extract_slice %arg0[2, 7, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_314 = tensor.collapse_shape %extracted_slice_313 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_315 = tensor.extract %collapsed_314[] : tensor<f32>
    %out_qubits_316:2 = quantum.custom "CNOT"() %out_qubits_289#1, %out_qubits_250#0 : !quantum.bit, !quantum.bit
    %83 = arith.extf %extracted_315 : f32 to f64
    %out_qubits_317 = quantum.custom "RZ"(%83) %out_qubits_316#0 : !quantum.bit
    %84 = arith.extf %extracted_312 : f32 to f64
    %out_qubits_318 = quantum.custom "RY"(%84) %out_qubits_317 : !quantum.bit
    %85 = arith.extf %extracted_309 : f32 to f64
    %out_qubits_319 = quantum.custom "RZ"(%85) %out_qubits_318 : !quantum.bit
    %extracted_slice_320 = tensor.extract_slice %arg0[2, 1, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_321 = tensor.collapse_shape %extracted_slice_320 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_322 = tensor.extract %collapsed_321[] : tensor<f32>
    %extracted_slice_323 = tensor.extract_slice %arg0[2, 1, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_324 = tensor.collapse_shape %extracted_slice_323 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_325 = tensor.extract %collapsed_324[] : tensor<f32>
    %extracted_slice_326 = tensor.extract_slice %arg0[2, 1, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_327 = tensor.collapse_shape %extracted_slice_326 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_328 = tensor.extract %collapsed_327[] : tensor<f32>
    %86 = arith.extf %extracted_328 : f32 to f64
    %out_qubits_329 = quantum.custom "RZ"(%86) %out_qubits_316#1 : !quantum.bit
    %87 = arith.extf %extracted_325 : f32 to f64
    %out_qubits_330 = quantum.custom "RY"(%87) %out_qubits_329 : !quantum.bit
    %88 = arith.extf %extracted_322 : f32 to f64
    %out_qubits_331 = quantum.custom "RZ"(%88) %out_qubits_330 : !quantum.bit
    %extracted_slice_332 = tensor.extract_slice %arg0[2, 4, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_333 = tensor.collapse_shape %extracted_slice_332 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_334 = tensor.extract %collapsed_333[] : tensor<f32>
    %extracted_slice_335 = tensor.extract_slice %arg0[2, 4, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_336 = tensor.collapse_shape %extracted_slice_335 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_337 = tensor.extract %collapsed_336[] : tensor<f32>
    %extracted_slice_338 = tensor.extract_slice %arg0[2, 4, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_339 = tensor.collapse_shape %extracted_slice_338 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_340 = tensor.extract %collapsed_339[] : tensor<f32>
    %89 = arith.extf %extracted_340 : f32 to f64
    %out_qubits_341 = quantum.custom "RZ"(%89) %out_qubits_200#0 : !quantum.bit
    %90 = arith.extf %extracted_337 : f32 to f64
    %out_qubits_342 = quantum.custom "RY"(%90) %out_qubits_341 : !quantum.bit
    %91 = arith.extf %extracted_334 : f32 to f64
    %out_qubits_343 = quantum.custom "RZ"(%91) %out_qubits_342 : !quantum.bit
    %out_qubits_344:2 = quantum.custom "CNOT"() %out_qubits_331, %out_qubits_343 : !quantum.bit, !quantum.bit
    %out_qubits_345:2 = quantum.custom "CNOT"() %out_qubits_344#1, %out_qubits_319 : !quantum.bit, !quantum.bit
    %92 = arith.extf %extracted_306 : f32 to f64
    %out_qubits_346 = quantum.custom "RZ"(%92) %out_qubits_345#0 : !quantum.bit
    %93 = arith.extf %extracted_303 : f32 to f64
    %out_qubits_347 = quantum.custom "RY"(%93) %out_qubits_346 : !quantum.bit
    %94 = arith.extf %extracted_300 : f32 to f64
    %out_qubits_348 = quantum.custom "RZ"(%94) %out_qubits_347 : !quantum.bit
    %out_qubits_349:2 = quantum.custom "CNOT"() %out_qubits_297, %out_qubits_348 : !quantum.bit, !quantum.bit
    %out_qubits_350:2 = quantum.custom "CNOT"() %out_qubits_349#1, %out_qubits_349#0 : !quantum.bit, !quantum.bit
    %extracted_slice_351 = tensor.extract_slice %arg0[3, 1, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_352 = tensor.collapse_shape %extracted_slice_351 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_353 = tensor.extract %collapsed_352[] : tensor<f32>
    %extracted_slice_354 = tensor.extract_slice %arg0[3, 1, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_355 = tensor.collapse_shape %extracted_slice_354 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_356 = tensor.extract %collapsed_355[] : tensor<f32>
    %extracted_slice_357 = tensor.extract_slice %arg0[3, 1, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_358 = tensor.collapse_shape %extracted_slice_357 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_359 = tensor.extract %collapsed_358[] : tensor<f32>
    %extracted_slice_360 = tensor.extract_slice %arg0[2, 6, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_361 = tensor.collapse_shape %extracted_slice_360 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_362 = tensor.extract %collapsed_361[] : tensor<f32>
    %extracted_slice_363 = tensor.extract_slice %arg0[2, 6, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_364 = tensor.collapse_shape %extracted_slice_363 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_365 = tensor.extract %collapsed_364[] : tensor<f32>
    %extracted_slice_366 = tensor.extract_slice %arg0[2, 6, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_367 = tensor.collapse_shape %extracted_slice_366 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_368 = tensor.extract %collapsed_367[] : tensor<f32>
    %95 = arith.extf %extracted_368 : f32 to f64
    %out_qubits_369 = quantum.custom "RZ"(%95) %out_qubits_201#0 : !quantum.bit
    %96 = arith.extf %extracted_365 : f32 to f64
    %out_qubits_370 = quantum.custom "RY"(%96) %out_qubits_369 : !quantum.bit
    %97 = arith.extf %extracted_362 : f32 to f64
    %out_qubits_371 = quantum.custom "RZ"(%97) %out_qubits_370 : !quantum.bit
    %out_qubits_372:2 = quantum.custom "CNOT"() %out_qubits_255#1, %out_qubits_371 : !quantum.bit, !quantum.bit
    %out_qubits_373:2 = quantum.custom "CNOT"() %out_qubits_372#1, %out_qubits_344#0 : !quantum.bit, !quantum.bit
    %98 = arith.extf %extracted_359 : f32 to f64
    %out_qubits_374 = quantum.custom "RZ"(%98) %out_qubits_373#1 : !quantum.bit
    %99 = arith.extf %extracted_356 : f32 to f64
    %out_qubits_375 = quantum.custom "RY"(%99) %out_qubits_374 : !quantum.bit
    %100 = arith.extf %extracted_353 : f32 to f64
    %out_qubits_376 = quantum.custom "RZ"(%100) %out_qubits_375 : !quantum.bit
    %extracted_slice_377 = tensor.extract_slice %arg0[3, 5, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_378 = tensor.collapse_shape %extracted_slice_377 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_379 = tensor.extract %collapsed_378[] : tensor<f32>
    %extracted_slice_380 = tensor.extract_slice %arg0[3, 5, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_381 = tensor.collapse_shape %extracted_slice_380 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_382 = tensor.extract %collapsed_381[] : tensor<f32>
    %extracted_slice_383 = tensor.extract_slice %arg0[3, 5, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_384 = tensor.collapse_shape %extracted_slice_383 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_385 = tensor.extract %collapsed_384[] : tensor<f32>
    %101 = arith.extf %extracted_385 : f32 to f64
    %out_qubits_386 = quantum.custom "RZ"(%101) %out_qubits_294#0 : !quantum.bit
    %102 = arith.extf %extracted_382 : f32 to f64
    %out_qubits_387 = quantum.custom "RY"(%102) %out_qubits_386 : !quantum.bit
    %103 = arith.extf %extracted_379 : f32 to f64
    %out_qubits_388 = quantum.custom "RZ"(%103) %out_qubits_387 : !quantum.bit
    %out_qubits_389:2 = quantum.custom "CNOT"() %out_qubits_376, %out_qubits_388 : !quantum.bit, !quantum.bit
    %out_qubits_390:2 = quantum.custom "CNOT"() %out_qubits_389#1, %out_qubits_389#0 : !quantum.bit, !quantum.bit
    %extracted_slice_391 = tensor.extract_slice %arg0[3, 2, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_392 = tensor.collapse_shape %extracted_slice_391 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_393 = tensor.extract %collapsed_392[] : tensor<f32>
    %extracted_slice_394 = tensor.extract_slice %arg0[3, 2, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_395 = tensor.collapse_shape %extracted_slice_394 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_396 = tensor.extract %collapsed_395[] : tensor<f32>
    %extracted_slice_397 = tensor.extract_slice %arg0[3, 2, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_398 = tensor.collapse_shape %extracted_slice_397 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_399 = tensor.extract %collapsed_398[] : tensor<f32>
    %out_qubits_400:2 = quantum.custom "CNOT"() %out_qubits_345#1, %out_qubits_293#0 : !quantum.bit, !quantum.bit
    %104 = arith.extf %extracted_399 : f32 to f64
    %out_qubits_401 = quantum.custom "RZ"(%104) %out_qubits_400#1 : !quantum.bit
    %105 = arith.extf %extracted_396 : f32 to f64
    %out_qubits_402 = quantum.custom "RY"(%105) %out_qubits_401 : !quantum.bit
    %106 = arith.extf %extracted_393 : f32 to f64
    %out_qubits_403 = quantum.custom "RZ"(%106) %out_qubits_402 : !quantum.bit
    %extracted_slice_404 = tensor.extract_slice %arg0[3, 6, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_405 = tensor.collapse_shape %extracted_slice_404 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_406 = tensor.extract %collapsed_405[] : tensor<f32>
    %extracted_slice_407 = tensor.extract_slice %arg0[3, 6, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_408 = tensor.collapse_shape %extracted_slice_407 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_409 = tensor.extract %collapsed_408[] : tensor<f32>
    %extracted_slice_410 = tensor.extract_slice %arg0[3, 6, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_411 = tensor.collapse_shape %extracted_slice_410 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_412 = tensor.extract %collapsed_411[] : tensor<f32>
    %107 = arith.extf %extracted_412 : f32 to f64
    %out_qubits_413 = quantum.custom "RZ"(%107) %out_qubits_373#0 : !quantum.bit
    %108 = arith.extf %extracted_409 : f32 to f64
    %out_qubits_414 = quantum.custom "RY"(%108) %out_qubits_413 : !quantum.bit
    %109 = arith.extf %extracted_406 : f32 to f64
    %out_qubits_415 = quantum.custom "RZ"(%109) %out_qubits_414 : !quantum.bit
    %out_qubits_416:2 = quantum.custom "CNOT"() %out_qubits_403, %out_qubits_415 : !quantum.bit, !quantum.bit
    %out_qubits_417:2 = quantum.custom "CNOT"() %out_qubits_416#1, %out_qubits_416#0 : !quantum.bit, !quantum.bit
    %extracted_slice_418 = tensor.extract_slice %arg0[3, 3, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_419 = tensor.collapse_shape %extracted_slice_418 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_420 = tensor.extract %collapsed_419[] : tensor<f32>
    %extracted_slice_421 = tensor.extract_slice %arg0[3, 3, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_422 = tensor.collapse_shape %extracted_slice_421 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_423 = tensor.extract %collapsed_422[] : tensor<f32>
    %extracted_slice_424 = tensor.extract_slice %arg0[3, 3, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_425 = tensor.collapse_shape %extracted_slice_424 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_426 = tensor.extract %collapsed_425[] : tensor<f32>
    %110 = arith.extf %extracted_426 : f32 to f64
    %out_qubits_427 = quantum.custom "RZ"(%110) %out_qubits_372#0 : !quantum.bit
    %111 = arith.extf %extracted_423 : f32 to f64
    %out_qubits_428 = quantum.custom "RY"(%111) %out_qubits_427 : !quantum.bit
    %112 = arith.extf %extracted_420 : f32 to f64
    %out_qubits_429 = quantum.custom "RZ"(%112) %out_qubits_428 : !quantum.bit
    %extracted_slice_430 = tensor.extract_slice %arg0[3, 7, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_431 = tensor.collapse_shape %extracted_slice_430 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_432 = tensor.extract %collapsed_431[] : tensor<f32>
    %extracted_slice_433 = tensor.extract_slice %arg0[3, 7, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_434 = tensor.collapse_shape %extracted_slice_433 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_435 = tensor.extract %collapsed_434[] : tensor<f32>
    %extracted_slice_436 = tensor.extract_slice %arg0[3, 7, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_437 = tensor.collapse_shape %extracted_slice_436 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_438 = tensor.extract %collapsed_437[] : tensor<f32>
    %113 = arith.extf %extracted_438 : f32 to f64
    %out_qubits_439 = quantum.custom "RZ"(%113) %out_qubits_400#0 : !quantum.bit
    %114 = arith.extf %extracted_435 : f32 to f64
    %out_qubits_440 = quantum.custom "RY"(%114) %out_qubits_439 : !quantum.bit
    %115 = arith.extf %extracted_432 : f32 to f64
    %out_qubits_441 = quantum.custom "RZ"(%115) %out_qubits_440 : !quantum.bit
    %out_qubits_442:2 = quantum.custom "CNOT"() %out_qubits_429, %out_qubits_441 : !quantum.bit, !quantum.bit
    %out_qubits_443:2 = quantum.custom "CNOT"() %out_qubits_442#1, %out_qubits_442#0 : !quantum.bit, !quantum.bit
    %116 = quantum.namedobs %out_qubits_350#1[ PauliZ] : !quantum.obs
    %117 = quantum.expval %116 : f64
    %118 = quantum.insert %3[ 0], %out_qubits_350#1 : !quantum.reg, !quantum.bit
    %119 = quantum.insert %118[ 1], %out_qubits_390#1 : !quantum.reg, !quantum.bit
    %120 = quantum.insert %119[ 2], %out_qubits_417#1 : !quantum.reg, !quantum.bit
    %121 = quantum.insert %120[ 3], %out_qubits_443#1 : !quantum.reg, !quantum.bit
    %122 = quantum.insert %121[ 4], %out_qubits_350#0 : !quantum.reg, !quantum.bit
    %123 = quantum.insert %122[ 5], %out_qubits_390#0 : !quantum.reg, !quantum.bit
    %124 = quantum.insert %123[ 6], %out_qubits_417#0 : !quantum.reg, !quantum.bit
    %125 = quantum.insert %124[ 7], %out_qubits_443#0 : !quantum.reg, !quantum.bit
    return %125, %117 : !quantum.reg, f64
  }
  func.func private @qnode_forward_0.pcount(%arg0: tensor<4x8x3xf32>, %arg1: tensor<8xf32>) -> index {
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
  func.func private @qnode_forward_0.quantum(%arg0: tensor<4x8x3xf32>, %arg1: tensor<8xf32>, %arg2: tensor<?xf64>) -> tensor<f64> attributes {gradient.qgrad = @qnode_forward_0.adjoint, passthrough = ["noinline"]} {
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
    %extracted = tensor.extract %arg2[%2] : tensor<?xf64>
    %out_qubits = quantum.custom "RY"(%extracted) %1 : !quantum.bit
    %4 = memref.load %alloca[] : memref<index>
    %5 = index.add %4, %idx1
    memref.store %5, %alloca[] : memref<index>
    %extracted_0 = tensor.extract %arg2[%4] : tensor<?xf64>
    %out_qubits_1 = quantum.custom "RZ"(%extracted_0) %out_qubits : !quantum.bit
    %6 = memref.load %alloca[] : memref<index>
    %7 = index.add %6, %idx1
    memref.store %7, %alloca[] : memref<index>
    %extracted_2 = tensor.extract %arg2[%6] : tensor<?xf64>
    %out_qubits_3 = quantum.custom "RY"(%extracted_2) %out_qubits_1 : !quantum.bit
    %8 = memref.load %alloca[] : memref<index>
    %9 = index.add %8, %idx1
    memref.store %9, %alloca[] : memref<index>
    %extracted_4 = tensor.extract %arg2[%8] : tensor<?xf64>
    %out_qubits_5 = quantum.custom "RZ"(%extracted_4) %out_qubits_3 : !quantum.bit
    %10 = quantum.extract %0[ 6] : !quantum.reg -> !quantum.bit
    %11 = memref.load %alloca[] : memref<index>
    %12 = index.add %11, %idx1
    memref.store %12, %alloca[] : memref<index>
    %extracted_6 = tensor.extract %arg2[%11] : tensor<?xf64>
    %out_qubits_7 = quantum.custom "RY"(%extracted_6) %10 : !quantum.bit
    %13 = memref.load %alloca[] : memref<index>
    %14 = index.add %13, %idx1
    memref.store %14, %alloca[] : memref<index>
    %extracted_8 = tensor.extract %arg2[%13] : tensor<?xf64>
    %out_qubits_9 = quantum.custom "RZ"(%extracted_8) %out_qubits_7 : !quantum.bit
    %15 = memref.load %alloca[] : memref<index>
    %16 = index.add %15, %idx1
    memref.store %16, %alloca[] : memref<index>
    %extracted_10 = tensor.extract %arg2[%15] : tensor<?xf64>
    %out_qubits_11 = quantum.custom "RY"(%extracted_10) %out_qubits_9 : !quantum.bit
    %17 = memref.load %alloca[] : memref<index>
    %18 = index.add %17, %idx1
    memref.store %18, %alloca[] : memref<index>
    %extracted_12 = tensor.extract %arg2[%17] : tensor<?xf64>
    %out_qubits_13 = quantum.custom "RZ"(%extracted_12) %out_qubits_11 : !quantum.bit
    %19 = quantum.extract %0[ 5] : !quantum.reg -> !quantum.bit
    %20 = memref.load %alloca[] : memref<index>
    %21 = index.add %20, %idx1
    memref.store %21, %alloca[] : memref<index>
    %extracted_14 = tensor.extract %arg2[%20] : tensor<?xf64>
    %out_qubits_15 = quantum.custom "RY"(%extracted_14) %19 : !quantum.bit
    %22 = memref.load %alloca[] : memref<index>
    %23 = index.add %22, %idx1
    memref.store %23, %alloca[] : memref<index>
    %extracted_16 = tensor.extract %arg2[%22] : tensor<?xf64>
    %out_qubits_17 = quantum.custom "RZ"(%extracted_16) %out_qubits_15 : !quantum.bit
    %24 = memref.load %alloca[] : memref<index>
    %25 = index.add %24, %idx1
    memref.store %25, %alloca[] : memref<index>
    %extracted_18 = tensor.extract %arg2[%24] : tensor<?xf64>
    %out_qubits_19 = quantum.custom "RY"(%extracted_18) %out_qubits_17 : !quantum.bit
    %26 = memref.load %alloca[] : memref<index>
    %27 = index.add %26, %idx1
    memref.store %27, %alloca[] : memref<index>
    %extracted_20 = tensor.extract %arg2[%26] : tensor<?xf64>
    %out_qubits_21 = quantum.custom "RZ"(%extracted_20) %out_qubits_19 : !quantum.bit
    %28 = quantum.extract %0[ 4] : !quantum.reg -> !quantum.bit
    %29 = memref.load %alloca[] : memref<index>
    %30 = index.add %29, %idx1
    memref.store %30, %alloca[] : memref<index>
    %extracted_22 = tensor.extract %arg2[%29] : tensor<?xf64>
    %out_qubits_23 = quantum.custom "RY"(%extracted_22) %28 : !quantum.bit
    %31 = memref.load %alloca[] : memref<index>
    %32 = index.add %31, %idx1
    memref.store %32, %alloca[] : memref<index>
    %extracted_24 = tensor.extract %arg2[%31] : tensor<?xf64>
    %out_qubits_25 = quantum.custom "RZ"(%extracted_24) %out_qubits_23 : !quantum.bit
    %33 = memref.load %alloca[] : memref<index>
    %34 = index.add %33, %idx1
    memref.store %34, %alloca[] : memref<index>
    %extracted_26 = tensor.extract %arg2[%33] : tensor<?xf64>
    %out_qubits_27 = quantum.custom "RY"(%extracted_26) %out_qubits_25 : !quantum.bit
    %35 = memref.load %alloca[] : memref<index>
    %36 = index.add %35, %idx1
    memref.store %36, %alloca[] : memref<index>
    %extracted_28 = tensor.extract %arg2[%35] : tensor<?xf64>
    %out_qubits_29 = quantum.custom "RZ"(%extracted_28) %out_qubits_27 : !quantum.bit
    %37 = quantum.extract %0[ 3] : !quantum.reg -> !quantum.bit
    %38 = memref.load %alloca[] : memref<index>
    %39 = index.add %38, %idx1
    memref.store %39, %alloca[] : memref<index>
    %extracted_30 = tensor.extract %arg2[%38] : tensor<?xf64>
    %out_qubits_31 = quantum.custom "RY"(%extracted_30) %37 : !quantum.bit
    %40 = memref.load %alloca[] : memref<index>
    %41 = index.add %40, %idx1
    memref.store %41, %alloca[] : memref<index>
    %extracted_32 = tensor.extract %arg2[%40] : tensor<?xf64>
    %out_qubits_33 = quantum.custom "RZ"(%extracted_32) %out_qubits_31 : !quantum.bit
    %42 = memref.load %alloca[] : memref<index>
    %43 = index.add %42, %idx1
    memref.store %43, %alloca[] : memref<index>
    %extracted_34 = tensor.extract %arg2[%42] : tensor<?xf64>
    %out_qubits_35 = quantum.custom "RY"(%extracted_34) %out_qubits_33 : !quantum.bit
    %44 = memref.load %alloca[] : memref<index>
    %45 = index.add %44, %idx1
    memref.store %45, %alloca[] : memref<index>
    %extracted_36 = tensor.extract %arg2[%44] : tensor<?xf64>
    %out_qubits_37 = quantum.custom "RZ"(%extracted_36) %out_qubits_35 : !quantum.bit
    %46 = quantum.extract %0[ 2] : !quantum.reg -> !quantum.bit
    %47 = memref.load %alloca[] : memref<index>
    %48 = index.add %47, %idx1
    memref.store %48, %alloca[] : memref<index>
    %extracted_38 = tensor.extract %arg2[%47] : tensor<?xf64>
    %out_qubits_39 = quantum.custom "RY"(%extracted_38) %46 : !quantum.bit
    %49 = memref.load %alloca[] : memref<index>
    %50 = index.add %49, %idx1
    memref.store %50, %alloca[] : memref<index>
    %extracted_40 = tensor.extract %arg2[%49] : tensor<?xf64>
    %out_qubits_41 = quantum.custom "RZ"(%extracted_40) %out_qubits_39 : !quantum.bit
    %51 = memref.load %alloca[] : memref<index>
    %52 = index.add %51, %idx1
    memref.store %52, %alloca[] : memref<index>
    %extracted_42 = tensor.extract %arg2[%51] : tensor<?xf64>
    %out_qubits_43 = quantum.custom "RY"(%extracted_42) %out_qubits_41 : !quantum.bit
    %53 = memref.load %alloca[] : memref<index>
    %54 = index.add %53, %idx1
    memref.store %54, %alloca[] : memref<index>
    %extracted_44 = tensor.extract %arg2[%53] : tensor<?xf64>
    %out_qubits_45 = quantum.custom "RZ"(%extracted_44) %out_qubits_43 : !quantum.bit
    %55 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %56 = memref.load %alloca[] : memref<index>
    %57 = index.add %56, %idx1
    memref.store %57, %alloca[] : memref<index>
    %extracted_46 = tensor.extract %arg2[%56] : tensor<?xf64>
    %out_qubits_47 = quantum.custom "RY"(%extracted_46) %55 : !quantum.bit
    %58 = memref.load %alloca[] : memref<index>
    %59 = index.add %58, %idx1
    memref.store %59, %alloca[] : memref<index>
    %extracted_48 = tensor.extract %arg2[%58] : tensor<?xf64>
    %out_qubits_49 = quantum.custom "RZ"(%extracted_48) %out_qubits_47 : !quantum.bit
    %60 = memref.load %alloca[] : memref<index>
    %61 = index.add %60, %idx1
    memref.store %61, %alloca[] : memref<index>
    %extracted_50 = tensor.extract %arg2[%60] : tensor<?xf64>
    %out_qubits_51 = quantum.custom "RY"(%extracted_50) %out_qubits_49 : !quantum.bit
    %62 = memref.load %alloca[] : memref<index>
    %63 = index.add %62, %idx1
    memref.store %63, %alloca[] : memref<index>
    %extracted_52 = tensor.extract %arg2[%62] : tensor<?xf64>
    %out_qubits_53 = quantum.custom "RZ"(%extracted_52) %out_qubits_51 : !quantum.bit
    %64 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    %65 = memref.load %alloca[] : memref<index>
    %66 = index.add %65, %idx1
    memref.store %66, %alloca[] : memref<index>
    %extracted_54 = tensor.extract %arg2[%65] : tensor<?xf64>
    %out_qubits_55 = quantum.custom "RY"(%extracted_54) %64 : !quantum.bit
    %67 = memref.load %alloca[] : memref<index>
    %68 = index.add %67, %idx1
    memref.store %68, %alloca[] : memref<index>
    %extracted_56 = tensor.extract %arg2[%67] : tensor<?xf64>
    %out_qubits_57 = quantum.custom "RZ"(%extracted_56) %out_qubits_55 : !quantum.bit
    %69 = memref.load %alloca[] : memref<index>
    %70 = index.add %69, %idx1
    memref.store %70, %alloca[] : memref<index>
    %extracted_58 = tensor.extract %arg2[%69] : tensor<?xf64>
    %out_qubits_59 = quantum.custom "RY"(%extracted_58) %out_qubits_57 : !quantum.bit
    %71 = memref.load %alloca[] : memref<index>
    %72 = index.add %71, %idx1
    memref.store %72, %alloca[] : memref<index>
    %extracted_60 = tensor.extract %arg2[%71] : tensor<?xf64>
    %out_qubits_61 = quantum.custom "RZ"(%extracted_60) %out_qubits_59 : !quantum.bit
    %out_qubits_62:2 = quantum.custom "CNOT"() %out_qubits_53, %out_qubits_61 : !quantum.bit, !quantum.bit
    %out_qubits_63:2 = quantum.custom "CNOT"() %out_qubits_62#1, %out_qubits_45 : !quantum.bit, !quantum.bit
    %out_qubits_64:2 = quantum.custom "CNOT"() %out_qubits_63#1, %out_qubits_37 : !quantum.bit, !quantum.bit
    %out_qubits_65:2 = quantum.custom "CNOT"() %out_qubits_64#1, %out_qubits_29 : !quantum.bit, !quantum.bit
    %out_qubits_66:2 = quantum.custom "CNOT"() %out_qubits_65#1, %out_qubits_21 : !quantum.bit, !quantum.bit
    %out_qubits_67:2 = quantum.custom "CNOT"() %out_qubits_66#1, %out_qubits_13 : !quantum.bit, !quantum.bit
    %out_qubits_68:2 = quantum.custom "CNOT"() %out_qubits_67#1, %out_qubits_5 : !quantum.bit, !quantum.bit
    %73 = memref.load %alloca[] : memref<index>
    %74 = index.add %73, %idx1
    memref.store %74, %alloca[] : memref<index>
    %extracted_69 = tensor.extract %arg2[%73] : tensor<?xf64>
    %out_qubits_70 = quantum.custom "RZ"(%extracted_69) %out_qubits_68#0 : !quantum.bit
    %75 = memref.load %alloca[] : memref<index>
    %76 = index.add %75, %idx1
    memref.store %76, %alloca[] : memref<index>
    %extracted_71 = tensor.extract %arg2[%75] : tensor<?xf64>
    %out_qubits_72 = quantum.custom "RY"(%extracted_71) %out_qubits_70 : !quantum.bit
    %77 = memref.load %alloca[] : memref<index>
    %78 = index.add %77, %idx1
    memref.store %78, %alloca[] : memref<index>
    %extracted_73 = tensor.extract %arg2[%77] : tensor<?xf64>
    %out_qubits_74 = quantum.custom "RZ"(%extracted_73) %out_qubits_72 : !quantum.bit
    %79 = memref.load %alloca[] : memref<index>
    %80 = index.add %79, %idx1
    memref.store %80, %alloca[] : memref<index>
    %extracted_75 = tensor.extract %arg2[%79] : tensor<?xf64>
    %out_qubits_76 = quantum.custom "RZ"(%extracted_75) %out_qubits_66#0 : !quantum.bit
    %81 = memref.load %alloca[] : memref<index>
    %82 = index.add %81, %idx1
    memref.store %82, %alloca[] : memref<index>
    %extracted_77 = tensor.extract %arg2[%81] : tensor<?xf64>
    %out_qubits_78 = quantum.custom "RY"(%extracted_77) %out_qubits_76 : !quantum.bit
    %83 = memref.load %alloca[] : memref<index>
    %84 = index.add %83, %idx1
    memref.store %84, %alloca[] : memref<index>
    %extracted_79 = tensor.extract %arg2[%83] : tensor<?xf64>
    %out_qubits_80 = quantum.custom "RZ"(%extracted_79) %out_qubits_78 : !quantum.bit
    %out_qubits_81:2 = quantum.custom "CNOT"() %out_qubits_68#1, %out_qubits_62#0 : !quantum.bit, !quantum.bit
    %85 = memref.load %alloca[] : memref<index>
    %86 = index.add %85, %idx1
    memref.store %86, %alloca[] : memref<index>
    %extracted_82 = tensor.extract %arg2[%85] : tensor<?xf64>
    %out_qubits_83 = quantum.custom "RZ"(%extracted_82) %out_qubits_81#1 : !quantum.bit
    %87 = memref.load %alloca[] : memref<index>
    %88 = index.add %87, %idx1
    memref.store %88, %alloca[] : memref<index>
    %extracted_84 = tensor.extract %arg2[%87] : tensor<?xf64>
    %out_qubits_85 = quantum.custom "RY"(%extracted_84) %out_qubits_83 : !quantum.bit
    %89 = memref.load %alloca[] : memref<index>
    %90 = index.add %89, %idx1
    memref.store %90, %alloca[] : memref<index>
    %extracted_86 = tensor.extract %arg2[%89] : tensor<?xf64>
    %out_qubits_87 = quantum.custom "RZ"(%extracted_86) %out_qubits_85 : !quantum.bit
    %91 = memref.load %alloca[] : memref<index>
    %92 = index.add %91, %idx1
    memref.store %92, %alloca[] : memref<index>
    %extracted_88 = tensor.extract %arg2[%91] : tensor<?xf64>
    %out_qubits_89 = quantum.custom "RZ"(%extracted_88) %out_qubits_64#0 : !quantum.bit
    %93 = memref.load %alloca[] : memref<index>
    %94 = index.add %93, %idx1
    memref.store %94, %alloca[] : memref<index>
    %extracted_90 = tensor.extract %arg2[%93] : tensor<?xf64>
    %out_qubits_91 = quantum.custom "RY"(%extracted_90) %out_qubits_89 : !quantum.bit
    %95 = memref.load %alloca[] : memref<index>
    %96 = index.add %95, %idx1
    memref.store %96, %alloca[] : memref<index>
    %extracted_92 = tensor.extract %arg2[%95] : tensor<?xf64>
    %out_qubits_93 = quantum.custom "RZ"(%extracted_92) %out_qubits_91 : !quantum.bit
    %out_qubits_94:2 = quantum.custom "CNOT"() %out_qubits_87, %out_qubits_93 : !quantum.bit, !quantum.bit
    %out_qubits_95:2 = quantum.custom "CNOT"() %out_qubits_94#1, %out_qubits_80 : !quantum.bit, !quantum.bit
    %out_qubits_96:2 = quantum.custom "CNOT"() %out_qubits_95#1, %out_qubits_74 : !quantum.bit, !quantum.bit
    %out_qubits_97:2 = quantum.custom "CNOT"() %out_qubits_96#1, %out_qubits_94#0 : !quantum.bit, !quantum.bit
    %97 = memref.load %alloca[] : memref<index>
    %98 = index.add %97, %idx1
    memref.store %98, %alloca[] : memref<index>
    %extracted_98 = tensor.extract %arg2[%97] : tensor<?xf64>
    %out_qubits_99 = quantum.custom "RZ"(%extracted_98) %out_qubits_97#1 : !quantum.bit
    %99 = memref.load %alloca[] : memref<index>
    %100 = index.add %99, %idx1
    memref.store %100, %alloca[] : memref<index>
    %extracted_100 = tensor.extract %arg2[%99] : tensor<?xf64>
    %out_qubits_101 = quantum.custom "RY"(%extracted_100) %out_qubits_99 : !quantum.bit
    %101 = memref.load %alloca[] : memref<index>
    %102 = index.add %101, %idx1
    memref.store %102, %alloca[] : memref<index>
    %extracted_102 = tensor.extract %arg2[%101] : tensor<?xf64>
    %out_qubits_103 = quantum.custom "RZ"(%extracted_102) %out_qubits_101 : !quantum.bit
    %103 = memref.load %alloca[] : memref<index>
    %104 = index.add %103, %idx1
    memref.store %104, %alloca[] : memref<index>
    %extracted_104 = tensor.extract %arg2[%103] : tensor<?xf64>
    %out_qubits_105 = quantum.custom "RZ"(%extracted_104) %out_qubits_67#0 : !quantum.bit
    %105 = memref.load %alloca[] : memref<index>
    %106 = index.add %105, %idx1
    memref.store %106, %alloca[] : memref<index>
    %extracted_106 = tensor.extract %arg2[%105] : tensor<?xf64>
    %out_qubits_107 = quantum.custom "RY"(%extracted_106) %out_qubits_105 : !quantum.bit
    %107 = memref.load %alloca[] : memref<index>
    %108 = index.add %107, %idx1
    memref.store %108, %alloca[] : memref<index>
    %extracted_108 = tensor.extract %arg2[%107] : tensor<?xf64>
    %out_qubits_109 = quantum.custom "RZ"(%extracted_108) %out_qubits_107 : !quantum.bit
    %109 = memref.load %alloca[] : memref<index>
    %110 = index.add %109, %idx1
    memref.store %110, %alloca[] : memref<index>
    %extracted_110 = tensor.extract %arg2[%109] : tensor<?xf64>
    %out_qubits_111 = quantum.custom "RZ"(%extracted_110) %out_qubits_63#0 : !quantum.bit
    %111 = memref.load %alloca[] : memref<index>
    %112 = index.add %111, %idx1
    memref.store %112, %alloca[] : memref<index>
    %extracted_112 = tensor.extract %arg2[%111] : tensor<?xf64>
    %out_qubits_113 = quantum.custom "RY"(%extracted_112) %out_qubits_111 : !quantum.bit
    %113 = memref.load %alloca[] : memref<index>
    %114 = index.add %113, %idx1
    memref.store %114, %alloca[] : memref<index>
    %extracted_114 = tensor.extract %arg2[%113] : tensor<?xf64>
    %out_qubits_115 = quantum.custom "RZ"(%extracted_114) %out_qubits_113 : !quantum.bit
    %115 = memref.load %alloca[] : memref<index>
    %116 = index.add %115, %idx1
    memref.store %116, %alloca[] : memref<index>
    %extracted_116 = tensor.extract %arg2[%115] : tensor<?xf64>
    %out_qubits_117 = quantum.custom "RZ"(%extracted_116) %out_qubits_65#0 : !quantum.bit
    %117 = memref.load %alloca[] : memref<index>
    %118 = index.add %117, %idx1
    memref.store %118, %alloca[] : memref<index>
    %extracted_118 = tensor.extract %arg2[%117] : tensor<?xf64>
    %out_qubits_119 = quantum.custom "RY"(%extracted_118) %out_qubits_117 : !quantum.bit
    %119 = memref.load %alloca[] : memref<index>
    %120 = index.add %119, %idx1
    memref.store %120, %alloca[] : memref<index>
    %extracted_120 = tensor.extract %arg2[%119] : tensor<?xf64>
    %out_qubits_121 = quantum.custom "RZ"(%extracted_120) %out_qubits_119 : !quantum.bit
    %out_qubits_122:2 = quantum.custom "CNOT"() %out_qubits_115, %out_qubits_121 : !quantum.bit, !quantum.bit
    %out_qubits_123:2 = quantum.custom "CNOT"() %out_qubits_122#1, %out_qubits_109 : !quantum.bit, !quantum.bit
    %121 = memref.load %alloca[] : memref<index>
    %122 = index.add %121, %idx1
    memref.store %122, %alloca[] : memref<index>
    %extracted_124 = tensor.extract %arg2[%121] : tensor<?xf64>
    %out_qubits_125 = quantum.custom "RZ"(%extracted_124) %out_qubits_123#0 : !quantum.bit
    %123 = memref.load %alloca[] : memref<index>
    %124 = index.add %123, %idx1
    memref.store %124, %alloca[] : memref<index>
    %extracted_126 = tensor.extract %arg2[%123] : tensor<?xf64>
    %out_qubits_127 = quantum.custom "RY"(%extracted_126) %out_qubits_125 : !quantum.bit
    %125 = memref.load %alloca[] : memref<index>
    %126 = index.add %125, %idx1
    memref.store %126, %alloca[] : memref<index>
    %extracted_128 = tensor.extract %arg2[%125] : tensor<?xf64>
    %out_qubits_129 = quantum.custom "RZ"(%extracted_128) %out_qubits_127 : !quantum.bit
    %out_qubits_130:2 = quantum.custom "CNOT"() %out_qubits_103, %out_qubits_129 : !quantum.bit, !quantum.bit
    %127 = memref.load %alloca[] : memref<index>
    %128 = index.add %127, %idx1
    memref.store %128, %alloca[] : memref<index>
    %extracted_131 = tensor.extract %arg2[%127] : tensor<?xf64>
    %out_qubits_132 = quantum.custom "RZ"(%extracted_131) %out_qubits_95#0 : !quantum.bit
    %129 = memref.load %alloca[] : memref<index>
    %130 = index.add %129, %idx1
    memref.store %130, %alloca[] : memref<index>
    %extracted_133 = tensor.extract %arg2[%129] : tensor<?xf64>
    %out_qubits_134 = quantum.custom "RY"(%extracted_133) %out_qubits_132 : !quantum.bit
    %131 = memref.load %alloca[] : memref<index>
    %132 = index.add %131, %idx1
    memref.store %132, %alloca[] : memref<index>
    %extracted_135 = tensor.extract %arg2[%131] : tensor<?xf64>
    %out_qubits_136 = quantum.custom "RZ"(%extracted_135) %out_qubits_134 : !quantum.bit
    %133 = memref.load %alloca[] : memref<index>
    %134 = index.add %133, %idx1
    memref.store %134, %alloca[] : memref<index>
    %extracted_137 = tensor.extract %arg2[%133] : tensor<?xf64>
    %out_qubits_138 = quantum.custom "RZ"(%extracted_137) %out_qubits_81#0 : !quantum.bit
    %135 = memref.load %alloca[] : memref<index>
    %136 = index.add %135, %idx1
    memref.store %136, %alloca[] : memref<index>
    %extracted_139 = tensor.extract %arg2[%135] : tensor<?xf64>
    %out_qubits_140 = quantum.custom "RY"(%extracted_139) %out_qubits_138 : !quantum.bit
    %137 = memref.load %alloca[] : memref<index>
    %138 = index.add %137, %idx1
    memref.store %138, %alloca[] : memref<index>
    %extracted_141 = tensor.extract %arg2[%137] : tensor<?xf64>
    %out_qubits_142 = quantum.custom "RZ"(%extracted_141) %out_qubits_140 : !quantum.bit
    %out_qubits_143:2 = quantum.custom "CNOT"() %out_qubits_123#1, %out_qubits_142 : !quantum.bit, !quantum.bit
    %139 = memref.load %alloca[] : memref<index>
    %140 = index.add %139, %idx1
    memref.store %140, %alloca[] : memref<index>
    %extracted_144 = tensor.extract %arg2[%139] : tensor<?xf64>
    %out_qubits_145 = quantum.custom "RZ"(%extracted_144) %out_qubits_143#0 : !quantum.bit
    %141 = memref.load %alloca[] : memref<index>
    %142 = index.add %141, %idx1
    memref.store %142, %alloca[] : memref<index>
    %extracted_146 = tensor.extract %arg2[%141] : tensor<?xf64>
    %out_qubits_147 = quantum.custom "RY"(%extracted_146) %out_qubits_145 : !quantum.bit
    %143 = memref.load %alloca[] : memref<index>
    %144 = index.add %143, %idx1
    memref.store %144, %alloca[] : memref<index>
    %extracted_148 = tensor.extract %arg2[%143] : tensor<?xf64>
    %out_qubits_149 = quantum.custom "RZ"(%extracted_148) %out_qubits_147 : !quantum.bit
    %out_qubits_150:2 = quantum.custom "CNOT"() %out_qubits_136, %out_qubits_149 : !quantum.bit, !quantum.bit
    %out_qubits_151:2 = quantum.custom "CNOT"() %out_qubits_150#1, %out_qubits_130#0 : !quantum.bit, !quantum.bit
    %145 = memref.load %alloca[] : memref<index>
    %146 = index.add %145, %idx1
    memref.store %146, %alloca[] : memref<index>
    %extracted_152 = tensor.extract %arg2[%145] : tensor<?xf64>
    %out_qubits_153 = quantum.custom "RZ"(%extracted_152) %out_qubits_151#1 : !quantum.bit
    %147 = memref.load %alloca[] : memref<index>
    %148 = index.add %147, %idx1
    memref.store %148, %alloca[] : memref<index>
    %extracted_154 = tensor.extract %arg2[%147] : tensor<?xf64>
    %out_qubits_155 = quantum.custom "RY"(%extracted_154) %out_qubits_153 : !quantum.bit
    %149 = memref.load %alloca[] : memref<index>
    %150 = index.add %149, %idx1
    memref.store %150, %alloca[] : memref<index>
    %extracted_156 = tensor.extract %arg2[%149] : tensor<?xf64>
    %out_qubits_157 = quantum.custom "RZ"(%extracted_156) %out_qubits_155 : !quantum.bit
    %out_qubits_158:2 = quantum.custom "CNOT"() %out_qubits_143#1, %out_qubits_122#0 : !quantum.bit, !quantum.bit
    %151 = memref.load %alloca[] : memref<index>
    %152 = index.add %151, %idx1
    memref.store %152, %alloca[] : memref<index>
    %extracted_159 = tensor.extract %arg2[%151] : tensor<?xf64>
    %out_qubits_160 = quantum.custom "RZ"(%extracted_159) %out_qubits_158#0 : !quantum.bit
    %153 = memref.load %alloca[] : memref<index>
    %154 = index.add %153, %idx1
    memref.store %154, %alloca[] : memref<index>
    %extracted_161 = tensor.extract %arg2[%153] : tensor<?xf64>
    %out_qubits_162 = quantum.custom "RY"(%extracted_161) %out_qubits_160 : !quantum.bit
    %155 = memref.load %alloca[] : memref<index>
    %156 = index.add %155, %idx1
    memref.store %156, %alloca[] : memref<index>
    %extracted_163 = tensor.extract %arg2[%155] : tensor<?xf64>
    %out_qubits_164 = quantum.custom "RZ"(%extracted_163) %out_qubits_162 : !quantum.bit
    %157 = memref.load %alloca[] : memref<index>
    %158 = index.add %157, %idx1
    memref.store %158, %alloca[] : memref<index>
    %extracted_165 = tensor.extract %arg2[%157] : tensor<?xf64>
    %out_qubits_166 = quantum.custom "RZ"(%extracted_165) %out_qubits_158#1 : !quantum.bit
    %159 = memref.load %alloca[] : memref<index>
    %160 = index.add %159, %idx1
    memref.store %160, %alloca[] : memref<index>
    %extracted_167 = tensor.extract %arg2[%159] : tensor<?xf64>
    %out_qubits_168 = quantum.custom "RY"(%extracted_167) %out_qubits_166 : !quantum.bit
    %161 = memref.load %alloca[] : memref<index>
    %162 = index.add %161, %idx1
    memref.store %162, %alloca[] : memref<index>
    %extracted_169 = tensor.extract %arg2[%161] : tensor<?xf64>
    %out_qubits_170 = quantum.custom "RZ"(%extracted_169) %out_qubits_168 : !quantum.bit
    %163 = memref.load %alloca[] : memref<index>
    %164 = index.add %163, %idx1
    memref.store %164, %alloca[] : memref<index>
    %extracted_171 = tensor.extract %arg2[%163] : tensor<?xf64>
    %out_qubits_172 = quantum.custom "RZ"(%extracted_171) %out_qubits_96#0 : !quantum.bit
    %165 = memref.load %alloca[] : memref<index>
    %166 = index.add %165, %idx1
    memref.store %166, %alloca[] : memref<index>
    %extracted_173 = tensor.extract %arg2[%165] : tensor<?xf64>
    %out_qubits_174 = quantum.custom "RY"(%extracted_173) %out_qubits_172 : !quantum.bit
    %167 = memref.load %alloca[] : memref<index>
    %168 = index.add %167, %idx1
    memref.store %168, %alloca[] : memref<index>
    %extracted_175 = tensor.extract %arg2[%167] : tensor<?xf64>
    %out_qubits_176 = quantum.custom "RZ"(%extracted_175) %out_qubits_174 : !quantum.bit
    %out_qubits_177:2 = quantum.custom "CNOT"() %out_qubits_170, %out_qubits_176 : !quantum.bit, !quantum.bit
    %out_qubits_178:2 = quantum.custom "CNOT"() %out_qubits_177#1, %out_qubits_164 : !quantum.bit, !quantum.bit
    %169 = memref.load %alloca[] : memref<index>
    %170 = index.add %169, %idx1
    memref.store %170, %alloca[] : memref<index>
    %extracted_179 = tensor.extract %arg2[%169] : tensor<?xf64>
    %out_qubits_180 = quantum.custom "RZ"(%extracted_179) %out_qubits_178#0 : !quantum.bit
    %171 = memref.load %alloca[] : memref<index>
    %172 = index.add %171, %idx1
    memref.store %172, %alloca[] : memref<index>
    %extracted_181 = tensor.extract %arg2[%171] : tensor<?xf64>
    %out_qubits_182 = quantum.custom "RY"(%extracted_181) %out_qubits_180 : !quantum.bit
    %173 = memref.load %alloca[] : memref<index>
    %174 = index.add %173, %idx1
    memref.store %174, %alloca[] : memref<index>
    %extracted_183 = tensor.extract %arg2[%173] : tensor<?xf64>
    %out_qubits_184 = quantum.custom "RZ"(%extracted_183) %out_qubits_182 : !quantum.bit
    %out_qubits_185:2 = quantum.custom "CNOT"() %out_qubits_157, %out_qubits_184 : !quantum.bit, !quantum.bit
    %out_qubits_186:2 = quantum.custom "CNOT"() %out_qubits_185#1, %out_qubits_185#0 : !quantum.bit, !quantum.bit
    %175 = memref.load %alloca[] : memref<index>
    %176 = index.add %175, %idx1
    memref.store %176, %alloca[] : memref<index>
    %extracted_187 = tensor.extract %arg2[%175] : tensor<?xf64>
    %out_qubits_188 = quantum.custom "RZ"(%extracted_187) %out_qubits_97#0 : !quantum.bit
    %177 = memref.load %alloca[] : memref<index>
    %178 = index.add %177, %idx1
    memref.store %178, %alloca[] : memref<index>
    %extracted_189 = tensor.extract %arg2[%177] : tensor<?xf64>
    %out_qubits_190 = quantum.custom "RY"(%extracted_189) %out_qubits_188 : !quantum.bit
    %179 = memref.load %alloca[] : memref<index>
    %180 = index.add %179, %idx1
    memref.store %180, %alloca[] : memref<index>
    %extracted_191 = tensor.extract %arg2[%179] : tensor<?xf64>
    %out_qubits_192 = quantum.custom "RZ"(%extracted_191) %out_qubits_190 : !quantum.bit
    %out_qubits_193:2 = quantum.custom "CNOT"() %out_qubits_130#1, %out_qubits_192 : !quantum.bit, !quantum.bit
    %out_qubits_194:2 = quantum.custom "CNOT"() %out_qubits_193#1, %out_qubits_177#0 : !quantum.bit, !quantum.bit
    %181 = memref.load %alloca[] : memref<index>
    %182 = index.add %181, %idx1
    memref.store %182, %alloca[] : memref<index>
    %extracted_195 = tensor.extract %arg2[%181] : tensor<?xf64>
    %out_qubits_196 = quantum.custom "RZ"(%extracted_195) %out_qubits_194#1 : !quantum.bit
    %183 = memref.load %alloca[] : memref<index>
    %184 = index.add %183, %idx1
    memref.store %184, %alloca[] : memref<index>
    %extracted_197 = tensor.extract %arg2[%183] : tensor<?xf64>
    %out_qubits_198 = quantum.custom "RY"(%extracted_197) %out_qubits_196 : !quantum.bit
    %185 = memref.load %alloca[] : memref<index>
    %186 = index.add %185, %idx1
    memref.store %186, %alloca[] : memref<index>
    %extracted_199 = tensor.extract %arg2[%185] : tensor<?xf64>
    %out_qubits_200 = quantum.custom "RZ"(%extracted_199) %out_qubits_198 : !quantum.bit
    %187 = memref.load %alloca[] : memref<index>
    %188 = index.add %187, %idx1
    memref.store %188, %alloca[] : memref<index>
    %extracted_201 = tensor.extract %arg2[%187] : tensor<?xf64>
    %out_qubits_202 = quantum.custom "RZ"(%extracted_201) %out_qubits_151#0 : !quantum.bit
    %189 = memref.load %alloca[] : memref<index>
    %190 = index.add %189, %idx1
    memref.store %190, %alloca[] : memref<index>
    %extracted_203 = tensor.extract %arg2[%189] : tensor<?xf64>
    %out_qubits_204 = quantum.custom "RY"(%extracted_203) %out_qubits_202 : !quantum.bit
    %191 = memref.load %alloca[] : memref<index>
    %192 = index.add %191, %idx1
    memref.store %192, %alloca[] : memref<index>
    %extracted_205 = tensor.extract %arg2[%191] : tensor<?xf64>
    %out_qubits_206 = quantum.custom "RZ"(%extracted_205) %out_qubits_204 : !quantum.bit
    %out_qubits_207:2 = quantum.custom "CNOT"() %out_qubits_200, %out_qubits_206 : !quantum.bit, !quantum.bit
    %out_qubits_208:2 = quantum.custom "CNOT"() %out_qubits_207#1, %out_qubits_207#0 : !quantum.bit, !quantum.bit
    %out_qubits_209:2 = quantum.custom "CNOT"() %out_qubits_178#1, %out_qubits_150#0 : !quantum.bit, !quantum.bit
    %193 = memref.load %alloca[] : memref<index>
    %194 = index.add %193, %idx1
    memref.store %194, %alloca[] : memref<index>
    %extracted_210 = tensor.extract %arg2[%193] : tensor<?xf64>
    %out_qubits_211 = quantum.custom "RZ"(%extracted_210) %out_qubits_209#1 : !quantum.bit
    %195 = memref.load %alloca[] : memref<index>
    %196 = index.add %195, %idx1
    memref.store %196, %alloca[] : memref<index>
    %extracted_212 = tensor.extract %arg2[%195] : tensor<?xf64>
    %out_qubits_213 = quantum.custom "RY"(%extracted_212) %out_qubits_211 : !quantum.bit
    %197 = memref.load %alloca[] : memref<index>
    %198 = index.add %197, %idx1
    memref.store %198, %alloca[] : memref<index>
    %extracted_214 = tensor.extract %arg2[%197] : tensor<?xf64>
    %out_qubits_215 = quantum.custom "RZ"(%extracted_214) %out_qubits_213 : !quantum.bit
    %199 = memref.load %alloca[] : memref<index>
    %200 = index.add %199, %idx1
    memref.store %200, %alloca[] : memref<index>
    %extracted_216 = tensor.extract %arg2[%199] : tensor<?xf64>
    %out_qubits_217 = quantum.custom "RZ"(%extracted_216) %out_qubits_194#0 : !quantum.bit
    %201 = memref.load %alloca[] : memref<index>
    %202 = index.add %201, %idx1
    memref.store %202, %alloca[] : memref<index>
    %extracted_218 = tensor.extract %arg2[%201] : tensor<?xf64>
    %out_qubits_219 = quantum.custom "RY"(%extracted_218) %out_qubits_217 : !quantum.bit
    %203 = memref.load %alloca[] : memref<index>
    %204 = index.add %203, %idx1
    memref.store %204, %alloca[] : memref<index>
    %extracted_220 = tensor.extract %arg2[%203] : tensor<?xf64>
    %out_qubits_221 = quantum.custom "RZ"(%extracted_220) %out_qubits_219 : !quantum.bit
    %out_qubits_222:2 = quantum.custom "CNOT"() %out_qubits_215, %out_qubits_221 : !quantum.bit, !quantum.bit
    %out_qubits_223:2 = quantum.custom "CNOT"() %out_qubits_222#1, %out_qubits_222#0 : !quantum.bit, !quantum.bit
    %205 = memref.load %alloca[] : memref<index>
    %206 = index.add %205, %idx1
    memref.store %206, %alloca[] : memref<index>
    %extracted_224 = tensor.extract %arg2[%205] : tensor<?xf64>
    %out_qubits_225 = quantum.custom "RZ"(%extracted_224) %out_qubits_193#0 : !quantum.bit
    %207 = memref.load %alloca[] : memref<index>
    %208 = index.add %207, %idx1
    memref.store %208, %alloca[] : memref<index>
    %extracted_226 = tensor.extract %arg2[%207] : tensor<?xf64>
    %out_qubits_227 = quantum.custom "RY"(%extracted_226) %out_qubits_225 : !quantum.bit
    %209 = memref.load %alloca[] : memref<index>
    %210 = index.add %209, %idx1
    memref.store %210, %alloca[] : memref<index>
    %extracted_228 = tensor.extract %arg2[%209] : tensor<?xf64>
    %out_qubits_229 = quantum.custom "RZ"(%extracted_228) %out_qubits_227 : !quantum.bit
    %211 = memref.load %alloca[] : memref<index>
    %212 = index.add %211, %idx1
    memref.store %212, %alloca[] : memref<index>
    %extracted_230 = tensor.extract %arg2[%211] : tensor<?xf64>
    %out_qubits_231 = quantum.custom "RZ"(%extracted_230) %out_qubits_209#0 : !quantum.bit
    %213 = memref.load %alloca[] : memref<index>
    %214 = index.add %213, %idx1
    memref.store %214, %alloca[] : memref<index>
    %extracted_232 = tensor.extract %arg2[%213] : tensor<?xf64>
    %out_qubits_233 = quantum.custom "RY"(%extracted_232) %out_qubits_231 : !quantum.bit
    %215 = memref.load %alloca[] : memref<index>
    %216 = index.add %215, %idx1
    memref.store %216, %alloca[] : memref<index>
    %extracted_234 = tensor.extract %arg2[%215] : tensor<?xf64>
    %out_qubits_235 = quantum.custom "RZ"(%extracted_234) %out_qubits_233 : !quantum.bit
    %out_qubits_236:2 = quantum.custom "CNOT"() %out_qubits_229, %out_qubits_235 : !quantum.bit, !quantum.bit
    %out_qubits_237:2 = quantum.custom "CNOT"() %out_qubits_236#1, %out_qubits_236#0 : !quantum.bit, !quantum.bit
    %217 = quantum.namedobs %out_qubits_186#1[ PauliZ] : !quantum.obs
    %218 = quantum.expval %217 : f64
    %from_elements = tensor.from_elements %218 : tensor<f64>
    %219 = quantum.insert %0[ 0], %out_qubits_186#1 : !quantum.reg, !quantum.bit
    %220 = quantum.insert %219[ 1], %out_qubits_208#1 : !quantum.reg, !quantum.bit
    %221 = quantum.insert %220[ 2], %out_qubits_223#1 : !quantum.reg, !quantum.bit
    %222 = quantum.insert %221[ 3], %out_qubits_237#1 : !quantum.reg, !quantum.bit
    %223 = quantum.insert %222[ 4], %out_qubits_186#0 : !quantum.reg, !quantum.bit
    %224 = quantum.insert %223[ 5], %out_qubits_208#0 : !quantum.reg, !quantum.bit
    %225 = quantum.insert %224[ 6], %out_qubits_223#0 : !quantum.reg, !quantum.bit
    %226 = quantum.insert %225[ 7], %out_qubits_237#0 : !quantum.reg, !quantum.bit
    quantum.dealloc %226 : !quantum.reg
    quantum.device_release
    return %from_elements : tensor<f64>
  }
  func.func private @qnode_forward_0.preprocess(%arg0: tensor<4x8x3xf32>, %arg1: tensor<8xf32>, %arg2: index) -> tensor<f64> {
    %cst = arith.constant dense<3.14159274> : tensor<f32>
    %idx1 = index.constant 1
    %idx0 = index.constant 0
    %alloc = memref.alloc(%arg2) : memref<?xf64>
    %0 = bufferization.to_tensor %alloc restrict : memref<?xf64> to tensor<?xf64>
    %alloca = memref.alloca() : memref<index>
    memref.store %idx0, %alloca[] : memref<index>
    %extracted_slice = tensor.extract_slice %arg0[3, 0, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed = tensor.collapse_shape %extracted_slice [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted = tensor.extract %collapsed[] : tensor<f32>
    %extracted_slice_0 = tensor.extract_slice %arg0[3, 0, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_1 = tensor.collapse_shape %extracted_slice_0 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_2 = tensor.extract %collapsed_1[] : tensor<f32>
    %extracted_slice_3 = tensor.extract_slice %arg0[3, 0, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_4 = tensor.collapse_shape %extracted_slice_3 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_5 = tensor.extract %collapsed_4[] : tensor<f32>
    %extracted_slice_6 = tensor.extract_slice %arg0[2, 0, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_7 = tensor.collapse_shape %extracted_slice_6 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_8 = tensor.extract %collapsed_7[] : tensor<f32>
    %extracted_slice_9 = tensor.extract_slice %arg0[2, 0, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_10 = tensor.collapse_shape %extracted_slice_9 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_11 = tensor.extract %collapsed_10[] : tensor<f32>
    %extracted_slice_12 = tensor.extract_slice %arg0[2, 0, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_13 = tensor.collapse_shape %extracted_slice_12 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_14 = tensor.extract %collapsed_13[] : tensor<f32>
    %extracted_slice_15 = tensor.extract_slice %arg0[1, 6, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_16 = tensor.collapse_shape %extracted_slice_15 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_17 = tensor.extract %collapsed_16[] : tensor<f32>
    %extracted_slice_18 = tensor.extract_slice %arg0[1, 6, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_19 = tensor.collapse_shape %extracted_slice_18 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_20 = tensor.extract %collapsed_19[] : tensor<f32>
    %extracted_slice_21 = tensor.extract_slice %arg0[1, 6, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_22 = tensor.collapse_shape %extracted_slice_21 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_23 = tensor.extract %collapsed_22[] : tensor<f32>
    %extracted_slice_24 = tensor.extract_slice %arg0[0, 7, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_25 = tensor.collapse_shape %extracted_slice_24 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_26 = tensor.extract %collapsed_25[] : tensor<f32>
    %extracted_slice_27 = tensor.extract_slice %arg0[0, 7, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_28 = tensor.collapse_shape %extracted_slice_27 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_29 = tensor.extract %collapsed_28[] : tensor<f32>
    %extracted_slice_30 = tensor.extract_slice %arg0[0, 7, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_31 = tensor.collapse_shape %extracted_slice_30 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_32 = tensor.extract %collapsed_31[] : tensor<f32>
    %1 = tensor.empty() : tensor<8xf32>
    %2 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%cst : tensor<f32>) outs(%1 : tensor<8xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<8xf32>
    %3 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%2, %arg1 : tensor<8xf32>, tensor<8xf32>) outs(%1 : tensor<8xf32>) {
    ^bb0(%in: f32, %in_309: f32, %out: f32):
      %317 = arith.mulf %in, %in_309 : f32
      linalg.yield %317 : f32
    } -> tensor<8xf32>
    %extracted_slice_33 = tensor.extract_slice %3[7] [1] [1] : tensor<8xf32> to tensor<1xf32>
    %collapsed_34 = tensor.collapse_shape %extracted_slice_33 [] : tensor<1xf32> into tensor<f32>
    %extracted_35 = tensor.extract %collapsed_34[] : tensor<f32>
    %4 = arith.extf %extracted_35 : f32 to f64
    %5 = memref.load %alloca[] : memref<index>
    memref.store %4, %alloc[%5] : memref<?xf64>
    %6 = index.add %5, %idx1
    memref.store %6, %alloca[] : memref<index>
    %7 = arith.extf %extracted_32 : f32 to f64
    %8 = memref.load %alloca[] : memref<index>
    memref.store %7, %alloc[%8] : memref<?xf64>
    %9 = index.add %8, %idx1
    memref.store %9, %alloca[] : memref<index>
    %10 = arith.extf %extracted_29 : f32 to f64
    %11 = memref.load %alloca[] : memref<index>
    memref.store %10, %alloc[%11] : memref<?xf64>
    %12 = index.add %11, %idx1
    memref.store %12, %alloca[] : memref<index>
    %13 = arith.extf %extracted_26 : f32 to f64
    %14 = memref.load %alloca[] : memref<index>
    memref.store %13, %alloc[%14] : memref<?xf64>
    %15 = index.add %14, %idx1
    memref.store %15, %alloca[] : memref<index>
    %extracted_slice_36 = tensor.extract_slice %arg0[0, 6, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_37 = tensor.collapse_shape %extracted_slice_36 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_38 = tensor.extract %collapsed_37[] : tensor<f32>
    %extracted_slice_39 = tensor.extract_slice %arg0[0, 6, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_40 = tensor.collapse_shape %extracted_slice_39 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_41 = tensor.extract %collapsed_40[] : tensor<f32>
    %extracted_slice_42 = tensor.extract_slice %arg0[0, 6, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_43 = tensor.collapse_shape %extracted_slice_42 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_44 = tensor.extract %collapsed_43[] : tensor<f32>
    %extracted_slice_45 = tensor.extract_slice %3[6] [1] [1] : tensor<8xf32> to tensor<1xf32>
    %collapsed_46 = tensor.collapse_shape %extracted_slice_45 [] : tensor<1xf32> into tensor<f32>
    %extracted_47 = tensor.extract %collapsed_46[] : tensor<f32>
    %16 = arith.extf %extracted_47 : f32 to f64
    %17 = memref.load %alloca[] : memref<index>
    memref.store %16, %alloc[%17] : memref<?xf64>
    %18 = index.add %17, %idx1
    memref.store %18, %alloca[] : memref<index>
    %19 = arith.extf %extracted_44 : f32 to f64
    %20 = memref.load %alloca[] : memref<index>
    memref.store %19, %alloc[%20] : memref<?xf64>
    %21 = index.add %20, %idx1
    memref.store %21, %alloca[] : memref<index>
    %22 = arith.extf %extracted_41 : f32 to f64
    %23 = memref.load %alloca[] : memref<index>
    memref.store %22, %alloc[%23] : memref<?xf64>
    %24 = index.add %23, %idx1
    memref.store %24, %alloca[] : memref<index>
    %25 = arith.extf %extracted_38 : f32 to f64
    %26 = memref.load %alloca[] : memref<index>
    memref.store %25, %alloc[%26] : memref<?xf64>
    %27 = index.add %26, %idx1
    memref.store %27, %alloca[] : memref<index>
    %extracted_slice_48 = tensor.extract_slice %arg0[0, 5, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_49 = tensor.collapse_shape %extracted_slice_48 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_50 = tensor.extract %collapsed_49[] : tensor<f32>
    %extracted_slice_51 = tensor.extract_slice %arg0[0, 5, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_52 = tensor.collapse_shape %extracted_slice_51 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_53 = tensor.extract %collapsed_52[] : tensor<f32>
    %extracted_slice_54 = tensor.extract_slice %arg0[0, 5, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_55 = tensor.collapse_shape %extracted_slice_54 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_56 = tensor.extract %collapsed_55[] : tensor<f32>
    %extracted_slice_57 = tensor.extract_slice %3[5] [1] [1] : tensor<8xf32> to tensor<1xf32>
    %collapsed_58 = tensor.collapse_shape %extracted_slice_57 [] : tensor<1xf32> into tensor<f32>
    %extracted_59 = tensor.extract %collapsed_58[] : tensor<f32>
    %28 = arith.extf %extracted_59 : f32 to f64
    %29 = memref.load %alloca[] : memref<index>
    memref.store %28, %alloc[%29] : memref<?xf64>
    %30 = index.add %29, %idx1
    memref.store %30, %alloca[] : memref<index>
    %31 = arith.extf %extracted_56 : f32 to f64
    %32 = memref.load %alloca[] : memref<index>
    memref.store %31, %alloc[%32] : memref<?xf64>
    %33 = index.add %32, %idx1
    memref.store %33, %alloca[] : memref<index>
    %34 = arith.extf %extracted_53 : f32 to f64
    %35 = memref.load %alloca[] : memref<index>
    memref.store %34, %alloc[%35] : memref<?xf64>
    %36 = index.add %35, %idx1
    memref.store %36, %alloca[] : memref<index>
    %37 = arith.extf %extracted_50 : f32 to f64
    %38 = memref.load %alloca[] : memref<index>
    memref.store %37, %alloc[%38] : memref<?xf64>
    %39 = index.add %38, %idx1
    memref.store %39, %alloca[] : memref<index>
    %extracted_slice_60 = tensor.extract_slice %arg0[0, 4, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_61 = tensor.collapse_shape %extracted_slice_60 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_62 = tensor.extract %collapsed_61[] : tensor<f32>
    %extracted_slice_63 = tensor.extract_slice %arg0[0, 4, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_64 = tensor.collapse_shape %extracted_slice_63 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_65 = tensor.extract %collapsed_64[] : tensor<f32>
    %extracted_slice_66 = tensor.extract_slice %arg0[0, 4, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_67 = tensor.collapse_shape %extracted_slice_66 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_68 = tensor.extract %collapsed_67[] : tensor<f32>
    %extracted_slice_69 = tensor.extract_slice %3[4] [1] [1] : tensor<8xf32> to tensor<1xf32>
    %collapsed_70 = tensor.collapse_shape %extracted_slice_69 [] : tensor<1xf32> into tensor<f32>
    %extracted_71 = tensor.extract %collapsed_70[] : tensor<f32>
    %40 = arith.extf %extracted_71 : f32 to f64
    %41 = memref.load %alloca[] : memref<index>
    memref.store %40, %alloc[%41] : memref<?xf64>
    %42 = index.add %41, %idx1
    memref.store %42, %alloca[] : memref<index>
    %43 = arith.extf %extracted_68 : f32 to f64
    %44 = memref.load %alloca[] : memref<index>
    memref.store %43, %alloc[%44] : memref<?xf64>
    %45 = index.add %44, %idx1
    memref.store %45, %alloca[] : memref<index>
    %46 = arith.extf %extracted_65 : f32 to f64
    %47 = memref.load %alloca[] : memref<index>
    memref.store %46, %alloc[%47] : memref<?xf64>
    %48 = index.add %47, %idx1
    memref.store %48, %alloca[] : memref<index>
    %49 = arith.extf %extracted_62 : f32 to f64
    %50 = memref.load %alloca[] : memref<index>
    memref.store %49, %alloc[%50] : memref<?xf64>
    %51 = index.add %50, %idx1
    memref.store %51, %alloca[] : memref<index>
    %extracted_slice_72 = tensor.extract_slice %arg0[0, 3, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_73 = tensor.collapse_shape %extracted_slice_72 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_74 = tensor.extract %collapsed_73[] : tensor<f32>
    %extracted_slice_75 = tensor.extract_slice %arg0[0, 3, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_76 = tensor.collapse_shape %extracted_slice_75 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_77 = tensor.extract %collapsed_76[] : tensor<f32>
    %extracted_slice_78 = tensor.extract_slice %arg0[0, 3, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_79 = tensor.collapse_shape %extracted_slice_78 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_80 = tensor.extract %collapsed_79[] : tensor<f32>
    %extracted_slice_81 = tensor.extract_slice %3[3] [1] [1] : tensor<8xf32> to tensor<1xf32>
    %collapsed_82 = tensor.collapse_shape %extracted_slice_81 [] : tensor<1xf32> into tensor<f32>
    %extracted_83 = tensor.extract %collapsed_82[] : tensor<f32>
    %52 = arith.extf %extracted_83 : f32 to f64
    %53 = memref.load %alloca[] : memref<index>
    memref.store %52, %alloc[%53] : memref<?xf64>
    %54 = index.add %53, %idx1
    memref.store %54, %alloca[] : memref<index>
    %55 = arith.extf %extracted_80 : f32 to f64
    %56 = memref.load %alloca[] : memref<index>
    memref.store %55, %alloc[%56] : memref<?xf64>
    %57 = index.add %56, %idx1
    memref.store %57, %alloca[] : memref<index>
    %58 = arith.extf %extracted_77 : f32 to f64
    %59 = memref.load %alloca[] : memref<index>
    memref.store %58, %alloc[%59] : memref<?xf64>
    %60 = index.add %59, %idx1
    memref.store %60, %alloca[] : memref<index>
    %61 = arith.extf %extracted_74 : f32 to f64
    %62 = memref.load %alloca[] : memref<index>
    memref.store %61, %alloc[%62] : memref<?xf64>
    %63 = index.add %62, %idx1
    memref.store %63, %alloca[] : memref<index>
    %extracted_slice_84 = tensor.extract_slice %arg0[0, 2, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_85 = tensor.collapse_shape %extracted_slice_84 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_86 = tensor.extract %collapsed_85[] : tensor<f32>
    %extracted_slice_87 = tensor.extract_slice %arg0[0, 2, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_88 = tensor.collapse_shape %extracted_slice_87 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_89 = tensor.extract %collapsed_88[] : tensor<f32>
    %extracted_slice_90 = tensor.extract_slice %arg0[0, 2, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_91 = tensor.collapse_shape %extracted_slice_90 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_92 = tensor.extract %collapsed_91[] : tensor<f32>
    %extracted_slice_93 = tensor.extract_slice %3[2] [1] [1] : tensor<8xf32> to tensor<1xf32>
    %collapsed_94 = tensor.collapse_shape %extracted_slice_93 [] : tensor<1xf32> into tensor<f32>
    %extracted_95 = tensor.extract %collapsed_94[] : tensor<f32>
    %64 = arith.extf %extracted_95 : f32 to f64
    %65 = memref.load %alloca[] : memref<index>
    memref.store %64, %alloc[%65] : memref<?xf64>
    %66 = index.add %65, %idx1
    memref.store %66, %alloca[] : memref<index>
    %67 = arith.extf %extracted_92 : f32 to f64
    %68 = memref.load %alloca[] : memref<index>
    memref.store %67, %alloc[%68] : memref<?xf64>
    %69 = index.add %68, %idx1
    memref.store %69, %alloca[] : memref<index>
    %70 = arith.extf %extracted_89 : f32 to f64
    %71 = memref.load %alloca[] : memref<index>
    memref.store %70, %alloc[%71] : memref<?xf64>
    %72 = index.add %71, %idx1
    memref.store %72, %alloca[] : memref<index>
    %73 = arith.extf %extracted_86 : f32 to f64
    %74 = memref.load %alloca[] : memref<index>
    memref.store %73, %alloc[%74] : memref<?xf64>
    %75 = index.add %74, %idx1
    memref.store %75, %alloca[] : memref<index>
    %extracted_slice_96 = tensor.extract_slice %arg0[0, 0, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_97 = tensor.collapse_shape %extracted_slice_96 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_98 = tensor.extract %collapsed_97[] : tensor<f32>
    %extracted_slice_99 = tensor.extract_slice %arg0[0, 0, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_100 = tensor.collapse_shape %extracted_slice_99 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_101 = tensor.extract %collapsed_100[] : tensor<f32>
    %extracted_slice_102 = tensor.extract_slice %arg0[0, 0, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_103 = tensor.collapse_shape %extracted_slice_102 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_104 = tensor.extract %collapsed_103[] : tensor<f32>
    %extracted_slice_105 = tensor.extract_slice %3[0] [1] [1] : tensor<8xf32> to tensor<1xf32>
    %collapsed_106 = tensor.collapse_shape %extracted_slice_105 [] : tensor<1xf32> into tensor<f32>
    %extracted_107 = tensor.extract %collapsed_106[] : tensor<f32>
    %76 = arith.extf %extracted_107 : f32 to f64
    %77 = memref.load %alloca[] : memref<index>
    memref.store %76, %alloc[%77] : memref<?xf64>
    %78 = index.add %77, %idx1
    memref.store %78, %alloca[] : memref<index>
    %79 = arith.extf %extracted_104 : f32 to f64
    %80 = memref.load %alloca[] : memref<index>
    memref.store %79, %alloc[%80] : memref<?xf64>
    %81 = index.add %80, %idx1
    memref.store %81, %alloca[] : memref<index>
    %82 = arith.extf %extracted_101 : f32 to f64
    %83 = memref.load %alloca[] : memref<index>
    memref.store %82, %alloc[%83] : memref<?xf64>
    %84 = index.add %83, %idx1
    memref.store %84, %alloca[] : memref<index>
    %85 = arith.extf %extracted_98 : f32 to f64
    %86 = memref.load %alloca[] : memref<index>
    memref.store %85, %alloc[%86] : memref<?xf64>
    %87 = index.add %86, %idx1
    memref.store %87, %alloca[] : memref<index>
    %extracted_slice_108 = tensor.extract_slice %arg0[0, 1, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_109 = tensor.collapse_shape %extracted_slice_108 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_110 = tensor.extract %collapsed_109[] : tensor<f32>
    %extracted_slice_111 = tensor.extract_slice %arg0[0, 1, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_112 = tensor.collapse_shape %extracted_slice_111 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_113 = tensor.extract %collapsed_112[] : tensor<f32>
    %extracted_slice_114 = tensor.extract_slice %arg0[0, 1, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_115 = tensor.collapse_shape %extracted_slice_114 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_116 = tensor.extract %collapsed_115[] : tensor<f32>
    %extracted_slice_117 = tensor.extract_slice %3[1] [1] [1] : tensor<8xf32> to tensor<1xf32>
    %collapsed_118 = tensor.collapse_shape %extracted_slice_117 [] : tensor<1xf32> into tensor<f32>
    %extracted_119 = tensor.extract %collapsed_118[] : tensor<f32>
    %88 = arith.extf %extracted_119 : f32 to f64
    %89 = memref.load %alloca[] : memref<index>
    memref.store %88, %alloc[%89] : memref<?xf64>
    %90 = index.add %89, %idx1
    memref.store %90, %alloca[] : memref<index>
    %91 = arith.extf %extracted_116 : f32 to f64
    %92 = memref.load %alloca[] : memref<index>
    memref.store %91, %alloc[%92] : memref<?xf64>
    %93 = index.add %92, %idx1
    memref.store %93, %alloca[] : memref<index>
    %94 = arith.extf %extracted_113 : f32 to f64
    %95 = memref.load %alloca[] : memref<index>
    memref.store %94, %alloc[%95] : memref<?xf64>
    %96 = index.add %95, %idx1
    memref.store %96, %alloca[] : memref<index>
    %97 = arith.extf %extracted_110 : f32 to f64
    %98 = memref.load %alloca[] : memref<index>
    memref.store %97, %alloc[%98] : memref<?xf64>
    %99 = index.add %98, %idx1
    memref.store %99, %alloca[] : memref<index>
    %100 = arith.extf %extracted_23 : f32 to f64
    %101 = memref.load %alloca[] : memref<index>
    memref.store %100, %alloc[%101] : memref<?xf64>
    %102 = index.add %101, %idx1
    memref.store %102, %alloca[] : memref<index>
    %103 = arith.extf %extracted_20 : f32 to f64
    %104 = memref.load %alloca[] : memref<index>
    memref.store %103, %alloc[%104] : memref<?xf64>
    %105 = index.add %104, %idx1
    memref.store %105, %alloca[] : memref<index>
    %106 = arith.extf %extracted_17 : f32 to f64
    %107 = memref.load %alloca[] : memref<index>
    memref.store %106, %alloc[%107] : memref<?xf64>
    %108 = index.add %107, %idx1
    memref.store %108, %alloca[] : memref<index>
    %extracted_slice_120 = tensor.extract_slice %arg0[1, 4, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_121 = tensor.collapse_shape %extracted_slice_120 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_122 = tensor.extract %collapsed_121[] : tensor<f32>
    %extracted_slice_123 = tensor.extract_slice %arg0[1, 4, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_124 = tensor.collapse_shape %extracted_slice_123 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_125 = tensor.extract %collapsed_124[] : tensor<f32>
    %extracted_slice_126 = tensor.extract_slice %arg0[1, 4, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_127 = tensor.collapse_shape %extracted_slice_126 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_128 = tensor.extract %collapsed_127[] : tensor<f32>
    %109 = arith.extf %extracted_128 : f32 to f64
    %110 = memref.load %alloca[] : memref<index>
    memref.store %109, %alloc[%110] : memref<?xf64>
    %111 = index.add %110, %idx1
    memref.store %111, %alloca[] : memref<index>
    %112 = arith.extf %extracted_125 : f32 to f64
    %113 = memref.load %alloca[] : memref<index>
    memref.store %112, %alloc[%113] : memref<?xf64>
    %114 = index.add %113, %idx1
    memref.store %114, %alloca[] : memref<index>
    %115 = arith.extf %extracted_122 : f32 to f64
    %116 = memref.load %alloca[] : memref<index>
    memref.store %115, %alloc[%116] : memref<?xf64>
    %117 = index.add %116, %idx1
    memref.store %117, %alloca[] : memref<index>
    %extracted_slice_129 = tensor.extract_slice %arg0[1, 0, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_130 = tensor.collapse_shape %extracted_slice_129 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_131 = tensor.extract %collapsed_130[] : tensor<f32>
    %extracted_slice_132 = tensor.extract_slice %arg0[1, 0, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_133 = tensor.collapse_shape %extracted_slice_132 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_134 = tensor.extract %collapsed_133[] : tensor<f32>
    %extracted_slice_135 = tensor.extract_slice %arg0[1, 0, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_136 = tensor.collapse_shape %extracted_slice_135 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_137 = tensor.extract %collapsed_136[] : tensor<f32>
    %118 = arith.extf %extracted_137 : f32 to f64
    %119 = memref.load %alloca[] : memref<index>
    memref.store %118, %alloc[%119] : memref<?xf64>
    %120 = index.add %119, %idx1
    memref.store %120, %alloca[] : memref<index>
    %121 = arith.extf %extracted_134 : f32 to f64
    %122 = memref.load %alloca[] : memref<index>
    memref.store %121, %alloc[%122] : memref<?xf64>
    %123 = index.add %122, %idx1
    memref.store %123, %alloca[] : memref<index>
    %124 = arith.extf %extracted_131 : f32 to f64
    %125 = memref.load %alloca[] : memref<index>
    memref.store %124, %alloc[%125] : memref<?xf64>
    %126 = index.add %125, %idx1
    memref.store %126, %alloca[] : memref<index>
    %extracted_slice_138 = tensor.extract_slice %arg0[1, 2, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_139 = tensor.collapse_shape %extracted_slice_138 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_140 = tensor.extract %collapsed_139[] : tensor<f32>
    %extracted_slice_141 = tensor.extract_slice %arg0[1, 2, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_142 = tensor.collapse_shape %extracted_slice_141 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_143 = tensor.extract %collapsed_142[] : tensor<f32>
    %extracted_slice_144 = tensor.extract_slice %arg0[1, 2, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_145 = tensor.collapse_shape %extracted_slice_144 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_146 = tensor.extract %collapsed_145[] : tensor<f32>
    %127 = arith.extf %extracted_146 : f32 to f64
    %128 = memref.load %alloca[] : memref<index>
    memref.store %127, %alloc[%128] : memref<?xf64>
    %129 = index.add %128, %idx1
    memref.store %129, %alloca[] : memref<index>
    %130 = arith.extf %extracted_143 : f32 to f64
    %131 = memref.load %alloca[] : memref<index>
    memref.store %130, %alloc[%131] : memref<?xf64>
    %132 = index.add %131, %idx1
    memref.store %132, %alloca[] : memref<index>
    %133 = arith.extf %extracted_140 : f32 to f64
    %134 = memref.load %alloca[] : memref<index>
    memref.store %133, %alloc[%134] : memref<?xf64>
    %135 = index.add %134, %idx1
    memref.store %135, %alloca[] : memref<index>
    %136 = arith.extf %extracted_14 : f32 to f64
    %137 = memref.load %alloca[] : memref<index>
    memref.store %136, %alloc[%137] : memref<?xf64>
    %138 = index.add %137, %idx1
    memref.store %138, %alloca[] : memref<index>
    %139 = arith.extf %extracted_11 : f32 to f64
    %140 = memref.load %alloca[] : memref<index>
    memref.store %139, %alloc[%140] : memref<?xf64>
    %141 = index.add %140, %idx1
    memref.store %141, %alloca[] : memref<index>
    %142 = arith.extf %extracted_8 : f32 to f64
    %143 = memref.load %alloca[] : memref<index>
    memref.store %142, %alloc[%143] : memref<?xf64>
    %144 = index.add %143, %idx1
    memref.store %144, %alloca[] : memref<index>
    %extracted_slice_147 = tensor.extract_slice %arg0[2, 3, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_148 = tensor.collapse_shape %extracted_slice_147 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_149 = tensor.extract %collapsed_148[] : tensor<f32>
    %extracted_slice_150 = tensor.extract_slice %arg0[2, 3, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_151 = tensor.collapse_shape %extracted_slice_150 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_152 = tensor.extract %collapsed_151[] : tensor<f32>
    %extracted_slice_153 = tensor.extract_slice %arg0[2, 3, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_154 = tensor.collapse_shape %extracted_slice_153 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_155 = tensor.extract %collapsed_154[] : tensor<f32>
    %extracted_slice_156 = tensor.extract_slice %arg0[1, 5, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_157 = tensor.collapse_shape %extracted_slice_156 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_158 = tensor.extract %collapsed_157[] : tensor<f32>
    %extracted_slice_159 = tensor.extract_slice %arg0[1, 5, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_160 = tensor.collapse_shape %extracted_slice_159 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_161 = tensor.extract %collapsed_160[] : tensor<f32>
    %extracted_slice_162 = tensor.extract_slice %arg0[1, 5, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_163 = tensor.collapse_shape %extracted_slice_162 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_164 = tensor.extract %collapsed_163[] : tensor<f32>
    %145 = arith.extf %extracted_164 : f32 to f64
    %146 = memref.load %alloca[] : memref<index>
    memref.store %145, %alloc[%146] : memref<?xf64>
    %147 = index.add %146, %idx1
    memref.store %147, %alloca[] : memref<index>
    %148 = arith.extf %extracted_161 : f32 to f64
    %149 = memref.load %alloca[] : memref<index>
    memref.store %148, %alloc[%149] : memref<?xf64>
    %150 = index.add %149, %idx1
    memref.store %150, %alloca[] : memref<index>
    %151 = arith.extf %extracted_158 : f32 to f64
    %152 = memref.load %alloca[] : memref<index>
    memref.store %151, %alloc[%152] : memref<?xf64>
    %153 = index.add %152, %idx1
    memref.store %153, %alloca[] : memref<index>
    %extracted_slice_165 = tensor.extract_slice %arg0[1, 1, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_166 = tensor.collapse_shape %extracted_slice_165 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_167 = tensor.extract %collapsed_166[] : tensor<f32>
    %extracted_slice_168 = tensor.extract_slice %arg0[1, 1, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_169 = tensor.collapse_shape %extracted_slice_168 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_170 = tensor.extract %collapsed_169[] : tensor<f32>
    %extracted_slice_171 = tensor.extract_slice %arg0[1, 1, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_172 = tensor.collapse_shape %extracted_slice_171 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_173 = tensor.extract %collapsed_172[] : tensor<f32>
    %154 = arith.extf %extracted_173 : f32 to f64
    %155 = memref.load %alloca[] : memref<index>
    memref.store %154, %alloc[%155] : memref<?xf64>
    %156 = index.add %155, %idx1
    memref.store %156, %alloca[] : memref<index>
    %157 = arith.extf %extracted_170 : f32 to f64
    %158 = memref.load %alloca[] : memref<index>
    memref.store %157, %alloc[%158] : memref<?xf64>
    %159 = index.add %158, %idx1
    memref.store %159, %alloca[] : memref<index>
    %160 = arith.extf %extracted_167 : f32 to f64
    %161 = memref.load %alloca[] : memref<index>
    memref.store %160, %alloc[%161] : memref<?xf64>
    %162 = index.add %161, %idx1
    memref.store %162, %alloca[] : memref<index>
    %extracted_slice_174 = tensor.extract_slice %arg0[1, 3, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_175 = tensor.collapse_shape %extracted_slice_174 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_176 = tensor.extract %collapsed_175[] : tensor<f32>
    %extracted_slice_177 = tensor.extract_slice %arg0[1, 3, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_178 = tensor.collapse_shape %extracted_slice_177 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_179 = tensor.extract %collapsed_178[] : tensor<f32>
    %extracted_slice_180 = tensor.extract_slice %arg0[1, 3, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_181 = tensor.collapse_shape %extracted_slice_180 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_182 = tensor.extract %collapsed_181[] : tensor<f32>
    %163 = arith.extf %extracted_182 : f32 to f64
    %164 = memref.load %alloca[] : memref<index>
    memref.store %163, %alloc[%164] : memref<?xf64>
    %165 = index.add %164, %idx1
    memref.store %165, %alloca[] : memref<index>
    %166 = arith.extf %extracted_179 : f32 to f64
    %167 = memref.load %alloca[] : memref<index>
    memref.store %166, %alloc[%167] : memref<?xf64>
    %168 = index.add %167, %idx1
    memref.store %168, %alloca[] : memref<index>
    %169 = arith.extf %extracted_176 : f32 to f64
    %170 = memref.load %alloca[] : memref<index>
    memref.store %169, %alloc[%170] : memref<?xf64>
    %171 = index.add %170, %idx1
    memref.store %171, %alloca[] : memref<index>
    %172 = arith.extf %extracted_155 : f32 to f64
    %173 = memref.load %alloca[] : memref<index>
    memref.store %172, %alloc[%173] : memref<?xf64>
    %174 = index.add %173, %idx1
    memref.store %174, %alloca[] : memref<index>
    %175 = arith.extf %extracted_152 : f32 to f64
    %176 = memref.load %alloca[] : memref<index>
    memref.store %175, %alloc[%176] : memref<?xf64>
    %177 = index.add %176, %idx1
    memref.store %177, %alloca[] : memref<index>
    %178 = arith.extf %extracted_149 : f32 to f64
    %179 = memref.load %alloca[] : memref<index>
    memref.store %178, %alloc[%179] : memref<?xf64>
    %180 = index.add %179, %idx1
    memref.store %180, %alloca[] : memref<index>
    %extracted_slice_183 = tensor.extract_slice %arg0[2, 2, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_184 = tensor.collapse_shape %extracted_slice_183 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_185 = tensor.extract %collapsed_184[] : tensor<f32>
    %extracted_slice_186 = tensor.extract_slice %arg0[2, 2, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_187 = tensor.collapse_shape %extracted_slice_186 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_188 = tensor.extract %collapsed_187[] : tensor<f32>
    %extracted_slice_189 = tensor.extract_slice %arg0[2, 2, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_190 = tensor.collapse_shape %extracted_slice_189 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_191 = tensor.extract %collapsed_190[] : tensor<f32>
    %181 = arith.extf %extracted_191 : f32 to f64
    %182 = memref.load %alloca[] : memref<index>
    memref.store %181, %alloc[%182] : memref<?xf64>
    %183 = index.add %182, %idx1
    memref.store %183, %alloca[] : memref<index>
    %184 = arith.extf %extracted_188 : f32 to f64
    %185 = memref.load %alloca[] : memref<index>
    memref.store %184, %alloc[%185] : memref<?xf64>
    %186 = index.add %185, %idx1
    memref.store %186, %alloca[] : memref<index>
    %187 = arith.extf %extracted_185 : f32 to f64
    %188 = memref.load %alloca[] : memref<index>
    memref.store %187, %alloc[%188] : memref<?xf64>
    %189 = index.add %188, %idx1
    memref.store %189, %alloca[] : memref<index>
    %extracted_slice_192 = tensor.extract_slice %arg0[2, 5, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_193 = tensor.collapse_shape %extracted_slice_192 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_194 = tensor.extract %collapsed_193[] : tensor<f32>
    %extracted_slice_195 = tensor.extract_slice %arg0[2, 5, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_196 = tensor.collapse_shape %extracted_slice_195 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_197 = tensor.extract %collapsed_196[] : tensor<f32>
    %extracted_slice_198 = tensor.extract_slice %arg0[2, 5, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_199 = tensor.collapse_shape %extracted_slice_198 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_200 = tensor.extract %collapsed_199[] : tensor<f32>
    %extracted_slice_201 = tensor.extract_slice %arg0[1, 7, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_202 = tensor.collapse_shape %extracted_slice_201 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_203 = tensor.extract %collapsed_202[] : tensor<f32>
    %extracted_slice_204 = tensor.extract_slice %arg0[1, 7, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_205 = tensor.collapse_shape %extracted_slice_204 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_206 = tensor.extract %collapsed_205[] : tensor<f32>
    %extracted_slice_207 = tensor.extract_slice %arg0[1, 7, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_208 = tensor.collapse_shape %extracted_slice_207 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_209 = tensor.extract %collapsed_208[] : tensor<f32>
    %190 = arith.extf %extracted_209 : f32 to f64
    %191 = memref.load %alloca[] : memref<index>
    memref.store %190, %alloc[%191] : memref<?xf64>
    %192 = index.add %191, %idx1
    memref.store %192, %alloca[] : memref<index>
    %193 = arith.extf %extracted_206 : f32 to f64
    %194 = memref.load %alloca[] : memref<index>
    memref.store %193, %alloc[%194] : memref<?xf64>
    %195 = index.add %194, %idx1
    memref.store %195, %alloca[] : memref<index>
    %196 = arith.extf %extracted_203 : f32 to f64
    %197 = memref.load %alloca[] : memref<index>
    memref.store %196, %alloc[%197] : memref<?xf64>
    %198 = index.add %197, %idx1
    memref.store %198, %alloca[] : memref<index>
    %199 = arith.extf %extracted_200 : f32 to f64
    %200 = memref.load %alloca[] : memref<index>
    memref.store %199, %alloc[%200] : memref<?xf64>
    %201 = index.add %200, %idx1
    memref.store %201, %alloca[] : memref<index>
    %202 = arith.extf %extracted_197 : f32 to f64
    %203 = memref.load %alloca[] : memref<index>
    memref.store %202, %alloc[%203] : memref<?xf64>
    %204 = index.add %203, %idx1
    memref.store %204, %alloca[] : memref<index>
    %205 = arith.extf %extracted_194 : f32 to f64
    %206 = memref.load %alloca[] : memref<index>
    memref.store %205, %alloc[%206] : memref<?xf64>
    %207 = index.add %206, %idx1
    memref.store %207, %alloca[] : memref<index>
    %208 = arith.extf %extracted_5 : f32 to f64
    %209 = memref.load %alloca[] : memref<index>
    memref.store %208, %alloc[%209] : memref<?xf64>
    %210 = index.add %209, %idx1
    memref.store %210, %alloca[] : memref<index>
    %211 = arith.extf %extracted_2 : f32 to f64
    %212 = memref.load %alloca[] : memref<index>
    memref.store %211, %alloc[%212] : memref<?xf64>
    %213 = index.add %212, %idx1
    memref.store %213, %alloca[] : memref<index>
    %214 = arith.extf %extracted : f32 to f64
    %215 = memref.load %alloca[] : memref<index>
    memref.store %214, %alloc[%215] : memref<?xf64>
    %216 = index.add %215, %idx1
    memref.store %216, %alloca[] : memref<index>
    %extracted_slice_210 = tensor.extract_slice %arg0[3, 4, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_211 = tensor.collapse_shape %extracted_slice_210 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_212 = tensor.extract %collapsed_211[] : tensor<f32>
    %extracted_slice_213 = tensor.extract_slice %arg0[3, 4, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_214 = tensor.collapse_shape %extracted_slice_213 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_215 = tensor.extract %collapsed_214[] : tensor<f32>
    %extracted_slice_216 = tensor.extract_slice %arg0[3, 4, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_217 = tensor.collapse_shape %extracted_slice_216 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_218 = tensor.extract %collapsed_217[] : tensor<f32>
    %extracted_slice_219 = tensor.extract_slice %arg0[2, 7, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_220 = tensor.collapse_shape %extracted_slice_219 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_221 = tensor.extract %collapsed_220[] : tensor<f32>
    %extracted_slice_222 = tensor.extract_slice %arg0[2, 7, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_223 = tensor.collapse_shape %extracted_slice_222 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_224 = tensor.extract %collapsed_223[] : tensor<f32>
    %extracted_slice_225 = tensor.extract_slice %arg0[2, 7, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_226 = tensor.collapse_shape %extracted_slice_225 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_227 = tensor.extract %collapsed_226[] : tensor<f32>
    %217 = arith.extf %extracted_227 : f32 to f64
    %218 = memref.load %alloca[] : memref<index>
    memref.store %217, %alloc[%218] : memref<?xf64>
    %219 = index.add %218, %idx1
    memref.store %219, %alloca[] : memref<index>
    %220 = arith.extf %extracted_224 : f32 to f64
    %221 = memref.load %alloca[] : memref<index>
    memref.store %220, %alloc[%221] : memref<?xf64>
    %222 = index.add %221, %idx1
    memref.store %222, %alloca[] : memref<index>
    %223 = arith.extf %extracted_221 : f32 to f64
    %224 = memref.load %alloca[] : memref<index>
    memref.store %223, %alloc[%224] : memref<?xf64>
    %225 = index.add %224, %idx1
    memref.store %225, %alloca[] : memref<index>
    %extracted_slice_228 = tensor.extract_slice %arg0[2, 1, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_229 = tensor.collapse_shape %extracted_slice_228 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_230 = tensor.extract %collapsed_229[] : tensor<f32>
    %extracted_slice_231 = tensor.extract_slice %arg0[2, 1, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_232 = tensor.collapse_shape %extracted_slice_231 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_233 = tensor.extract %collapsed_232[] : tensor<f32>
    %extracted_slice_234 = tensor.extract_slice %arg0[2, 1, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_235 = tensor.collapse_shape %extracted_slice_234 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_236 = tensor.extract %collapsed_235[] : tensor<f32>
    %226 = arith.extf %extracted_236 : f32 to f64
    %227 = memref.load %alloca[] : memref<index>
    memref.store %226, %alloc[%227] : memref<?xf64>
    %228 = index.add %227, %idx1
    memref.store %228, %alloca[] : memref<index>
    %229 = arith.extf %extracted_233 : f32 to f64
    %230 = memref.load %alloca[] : memref<index>
    memref.store %229, %alloc[%230] : memref<?xf64>
    %231 = index.add %230, %idx1
    memref.store %231, %alloca[] : memref<index>
    %232 = arith.extf %extracted_230 : f32 to f64
    %233 = memref.load %alloca[] : memref<index>
    memref.store %232, %alloc[%233] : memref<?xf64>
    %234 = index.add %233, %idx1
    memref.store %234, %alloca[] : memref<index>
    %extracted_slice_237 = tensor.extract_slice %arg0[2, 4, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_238 = tensor.collapse_shape %extracted_slice_237 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_239 = tensor.extract %collapsed_238[] : tensor<f32>
    %extracted_slice_240 = tensor.extract_slice %arg0[2, 4, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_241 = tensor.collapse_shape %extracted_slice_240 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_242 = tensor.extract %collapsed_241[] : tensor<f32>
    %extracted_slice_243 = tensor.extract_slice %arg0[2, 4, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_244 = tensor.collapse_shape %extracted_slice_243 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_245 = tensor.extract %collapsed_244[] : tensor<f32>
    %235 = arith.extf %extracted_245 : f32 to f64
    %236 = memref.load %alloca[] : memref<index>
    memref.store %235, %alloc[%236] : memref<?xf64>
    %237 = index.add %236, %idx1
    memref.store %237, %alloca[] : memref<index>
    %238 = arith.extf %extracted_242 : f32 to f64
    %239 = memref.load %alloca[] : memref<index>
    memref.store %238, %alloc[%239] : memref<?xf64>
    %240 = index.add %239, %idx1
    memref.store %240, %alloca[] : memref<index>
    %241 = arith.extf %extracted_239 : f32 to f64
    %242 = memref.load %alloca[] : memref<index>
    memref.store %241, %alloc[%242] : memref<?xf64>
    %243 = index.add %242, %idx1
    memref.store %243, %alloca[] : memref<index>
    %244 = arith.extf %extracted_218 : f32 to f64
    %245 = memref.load %alloca[] : memref<index>
    memref.store %244, %alloc[%245] : memref<?xf64>
    %246 = index.add %245, %idx1
    memref.store %246, %alloca[] : memref<index>
    %247 = arith.extf %extracted_215 : f32 to f64
    %248 = memref.load %alloca[] : memref<index>
    memref.store %247, %alloc[%248] : memref<?xf64>
    %249 = index.add %248, %idx1
    memref.store %249, %alloca[] : memref<index>
    %250 = arith.extf %extracted_212 : f32 to f64
    %251 = memref.load %alloca[] : memref<index>
    memref.store %250, %alloc[%251] : memref<?xf64>
    %252 = index.add %251, %idx1
    memref.store %252, %alloca[] : memref<index>
    %extracted_slice_246 = tensor.extract_slice %arg0[3, 1, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_247 = tensor.collapse_shape %extracted_slice_246 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_248 = tensor.extract %collapsed_247[] : tensor<f32>
    %extracted_slice_249 = tensor.extract_slice %arg0[3, 1, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_250 = tensor.collapse_shape %extracted_slice_249 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_251 = tensor.extract %collapsed_250[] : tensor<f32>
    %extracted_slice_252 = tensor.extract_slice %arg0[3, 1, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_253 = tensor.collapse_shape %extracted_slice_252 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_254 = tensor.extract %collapsed_253[] : tensor<f32>
    %extracted_slice_255 = tensor.extract_slice %arg0[2, 6, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_256 = tensor.collapse_shape %extracted_slice_255 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_257 = tensor.extract %collapsed_256[] : tensor<f32>
    %extracted_slice_258 = tensor.extract_slice %arg0[2, 6, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_259 = tensor.collapse_shape %extracted_slice_258 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_260 = tensor.extract %collapsed_259[] : tensor<f32>
    %extracted_slice_261 = tensor.extract_slice %arg0[2, 6, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_262 = tensor.collapse_shape %extracted_slice_261 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_263 = tensor.extract %collapsed_262[] : tensor<f32>
    %253 = arith.extf %extracted_263 : f32 to f64
    %254 = memref.load %alloca[] : memref<index>
    memref.store %253, %alloc[%254] : memref<?xf64>
    %255 = index.add %254, %idx1
    memref.store %255, %alloca[] : memref<index>
    %256 = arith.extf %extracted_260 : f32 to f64
    %257 = memref.load %alloca[] : memref<index>
    memref.store %256, %alloc[%257] : memref<?xf64>
    %258 = index.add %257, %idx1
    memref.store %258, %alloca[] : memref<index>
    %259 = arith.extf %extracted_257 : f32 to f64
    %260 = memref.load %alloca[] : memref<index>
    memref.store %259, %alloc[%260] : memref<?xf64>
    %261 = index.add %260, %idx1
    memref.store %261, %alloca[] : memref<index>
    %262 = arith.extf %extracted_254 : f32 to f64
    %263 = memref.load %alloca[] : memref<index>
    memref.store %262, %alloc[%263] : memref<?xf64>
    %264 = index.add %263, %idx1
    memref.store %264, %alloca[] : memref<index>
    %265 = arith.extf %extracted_251 : f32 to f64
    %266 = memref.load %alloca[] : memref<index>
    memref.store %265, %alloc[%266] : memref<?xf64>
    %267 = index.add %266, %idx1
    memref.store %267, %alloca[] : memref<index>
    %268 = arith.extf %extracted_248 : f32 to f64
    %269 = memref.load %alloca[] : memref<index>
    memref.store %268, %alloc[%269] : memref<?xf64>
    %270 = index.add %269, %idx1
    memref.store %270, %alloca[] : memref<index>
    %extracted_slice_264 = tensor.extract_slice %arg0[3, 5, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_265 = tensor.collapse_shape %extracted_slice_264 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_266 = tensor.extract %collapsed_265[] : tensor<f32>
    %extracted_slice_267 = tensor.extract_slice %arg0[3, 5, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_268 = tensor.collapse_shape %extracted_slice_267 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_269 = tensor.extract %collapsed_268[] : tensor<f32>
    %extracted_slice_270 = tensor.extract_slice %arg0[3, 5, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_271 = tensor.collapse_shape %extracted_slice_270 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_272 = tensor.extract %collapsed_271[] : tensor<f32>
    %271 = arith.extf %extracted_272 : f32 to f64
    %272 = memref.load %alloca[] : memref<index>
    memref.store %271, %alloc[%272] : memref<?xf64>
    %273 = index.add %272, %idx1
    memref.store %273, %alloca[] : memref<index>
    %274 = arith.extf %extracted_269 : f32 to f64
    %275 = memref.load %alloca[] : memref<index>
    memref.store %274, %alloc[%275] : memref<?xf64>
    %276 = index.add %275, %idx1
    memref.store %276, %alloca[] : memref<index>
    %277 = arith.extf %extracted_266 : f32 to f64
    %278 = memref.load %alloca[] : memref<index>
    memref.store %277, %alloc[%278] : memref<?xf64>
    %279 = index.add %278, %idx1
    memref.store %279, %alloca[] : memref<index>
    %extracted_slice_273 = tensor.extract_slice %arg0[3, 2, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_274 = tensor.collapse_shape %extracted_slice_273 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_275 = tensor.extract %collapsed_274[] : tensor<f32>
    %extracted_slice_276 = tensor.extract_slice %arg0[3, 2, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_277 = tensor.collapse_shape %extracted_slice_276 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_278 = tensor.extract %collapsed_277[] : tensor<f32>
    %extracted_slice_279 = tensor.extract_slice %arg0[3, 2, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_280 = tensor.collapse_shape %extracted_slice_279 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_281 = tensor.extract %collapsed_280[] : tensor<f32>
    %280 = arith.extf %extracted_281 : f32 to f64
    %281 = memref.load %alloca[] : memref<index>
    memref.store %280, %alloc[%281] : memref<?xf64>
    %282 = index.add %281, %idx1
    memref.store %282, %alloca[] : memref<index>
    %283 = arith.extf %extracted_278 : f32 to f64
    %284 = memref.load %alloca[] : memref<index>
    memref.store %283, %alloc[%284] : memref<?xf64>
    %285 = index.add %284, %idx1
    memref.store %285, %alloca[] : memref<index>
    %286 = arith.extf %extracted_275 : f32 to f64
    %287 = memref.load %alloca[] : memref<index>
    memref.store %286, %alloc[%287] : memref<?xf64>
    %288 = index.add %287, %idx1
    memref.store %288, %alloca[] : memref<index>
    %extracted_slice_282 = tensor.extract_slice %arg0[3, 6, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_283 = tensor.collapse_shape %extracted_slice_282 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_284 = tensor.extract %collapsed_283[] : tensor<f32>
    %extracted_slice_285 = tensor.extract_slice %arg0[3, 6, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_286 = tensor.collapse_shape %extracted_slice_285 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_287 = tensor.extract %collapsed_286[] : tensor<f32>
    %extracted_slice_288 = tensor.extract_slice %arg0[3, 6, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_289 = tensor.collapse_shape %extracted_slice_288 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_290 = tensor.extract %collapsed_289[] : tensor<f32>
    %289 = arith.extf %extracted_290 : f32 to f64
    %290 = memref.load %alloca[] : memref<index>
    memref.store %289, %alloc[%290] : memref<?xf64>
    %291 = index.add %290, %idx1
    memref.store %291, %alloca[] : memref<index>
    %292 = arith.extf %extracted_287 : f32 to f64
    %293 = memref.load %alloca[] : memref<index>
    memref.store %292, %alloc[%293] : memref<?xf64>
    %294 = index.add %293, %idx1
    memref.store %294, %alloca[] : memref<index>
    %295 = arith.extf %extracted_284 : f32 to f64
    %296 = memref.load %alloca[] : memref<index>
    memref.store %295, %alloc[%296] : memref<?xf64>
    %297 = index.add %296, %idx1
    memref.store %297, %alloca[] : memref<index>
    %extracted_slice_291 = tensor.extract_slice %arg0[3, 3, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_292 = tensor.collapse_shape %extracted_slice_291 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_293 = tensor.extract %collapsed_292[] : tensor<f32>
    %extracted_slice_294 = tensor.extract_slice %arg0[3, 3, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_295 = tensor.collapse_shape %extracted_slice_294 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_296 = tensor.extract %collapsed_295[] : tensor<f32>
    %extracted_slice_297 = tensor.extract_slice %arg0[3, 3, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_298 = tensor.collapse_shape %extracted_slice_297 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_299 = tensor.extract %collapsed_298[] : tensor<f32>
    %298 = arith.extf %extracted_299 : f32 to f64
    %299 = memref.load %alloca[] : memref<index>
    memref.store %298, %alloc[%299] : memref<?xf64>
    %300 = index.add %299, %idx1
    memref.store %300, %alloca[] : memref<index>
    %301 = arith.extf %extracted_296 : f32 to f64
    %302 = memref.load %alloca[] : memref<index>
    memref.store %301, %alloc[%302] : memref<?xf64>
    %303 = index.add %302, %idx1
    memref.store %303, %alloca[] : memref<index>
    %304 = arith.extf %extracted_293 : f32 to f64
    %305 = memref.load %alloca[] : memref<index>
    memref.store %304, %alloc[%305] : memref<?xf64>
    %306 = index.add %305, %idx1
    memref.store %306, %alloca[] : memref<index>
    %extracted_slice_300 = tensor.extract_slice %arg0[3, 7, 2] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_301 = tensor.collapse_shape %extracted_slice_300 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_302 = tensor.extract %collapsed_301[] : tensor<f32>
    %extracted_slice_303 = tensor.extract_slice %arg0[3, 7, 1] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_304 = tensor.collapse_shape %extracted_slice_303 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_305 = tensor.extract %collapsed_304[] : tensor<f32>
    %extracted_slice_306 = tensor.extract_slice %arg0[3, 7, 0] [1, 1, 1] [1, 1, 1] : tensor<4x8x3xf32> to tensor<1x1x1xf32>
    %collapsed_307 = tensor.collapse_shape %extracted_slice_306 [] : tensor<1x1x1xf32> into tensor<f32>
    %extracted_308 = tensor.extract %collapsed_307[] : tensor<f32>
    %307 = arith.extf %extracted_308 : f32 to f64
    %308 = memref.load %alloca[] : memref<index>
    memref.store %307, %alloc[%308] : memref<?xf64>
    %309 = index.add %308, %idx1
    memref.store %309, %alloca[] : memref<index>
    %310 = arith.extf %extracted_305 : f32 to f64
    %311 = memref.load %alloca[] : memref<index>
    memref.store %310, %alloc[%311] : memref<?xf64>
    %312 = index.add %311, %idx1
    memref.store %312, %alloca[] : memref<index>
    %313 = arith.extf %extracted_302 : f32 to f64
    %314 = memref.load %alloca[] : memref<index>
    memref.store %313, %alloc[%314] : memref<?xf64>
    %315 = index.add %314, %idx1
    memref.store %315, %alloca[] : memref<index>
    %316 = call @qnode_forward_0.quantum(%arg0, %arg1, %0) : (tensor<4x8x3xf32>, tensor<8xf32>, tensor<?xf64>) -> tensor<f64>
    return %316 : tensor<f64>
  }
  func.func private @qnode_forward_0.fullgrad0(%arg0: tensor<4x8x3xf32>, %arg1: tensor<8xf32>, %arg2: index) -> tensor<4x8x3xf32> {
    %cst = arith.constant 0.000000e+00 : f64
    %cst_0 = arith.constant 1.000000e+00 : f64
    %0 = tensor.empty() : tensor<f64>
    %1 = linalg.fill ins(%cst : f64) outs(%0 : tensor<f64>) -> tensor<f64>
    %inserted = tensor.insert %cst_0 into %1[] : tensor<f64>
    %gradients = gradient.backprop @qnode_forward_0.preprocess(%arg0, %arg1, %arg2) cotangents(%inserted : tensor<f64>) {diffArgIndices = dense<0> : tensor<1xi64>, keepValueResults = false, resultSegmentSizes = array<i32: 0, 1>} : (tensor<4x8x3xf32>, tensor<8xf32>, index) -> tensor<4x8x3xf32>
    return %gradients : tensor<4x8x3xf32>
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