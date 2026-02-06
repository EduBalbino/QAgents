module @train_epoch_compiled {
  func.func public @jit_train_epoch_compiled(%arg0: tensor<4x8x3xf32>, %arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<i32>, %arg4: tensor<2xi32>, %arg5: tensor<96x8xf32>, %arg6: tensor<96xf32>, %arg7: tensor<96xf32>) -> (tensor<4x8x3xf32>, tensor<f32>, tensor<f32>, tensor<i32>, tensor<2xi32>, tensor<2xf64>) attributes {llvm.emit_c_interface} {
    %cst = arith.constant 0.000000e+00 : f64
    %cst_0 = arith.constant dense<3.125000e-02> : tensor<f32>
    %cst_1 = arith.constant dense<0.00999999977> : tensor<f32>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<f32>
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c32_i64 = arith.constant 32 : i64
    %c0_i64 = arith.constant 0 : i64
    %c96_i64 = arith.constant 96 : i64
    %c95 = arith.constant 95 : index
    %cst_3 = arith.constant 0.00999999977 : f32
    %cst_4 = arith.constant 3.200000e+01 : f64
    %cst_5 = arith.constant 3.000000e+00 : f64
    %cst_6 = arith.constant 3.125000e-02 : f64
    %0 = call @_threefry_split(%arg4) : (tensor<2xi32>) -> tensor<2x2xi32>
    %extracted_slice = tensor.extract_slice %0[0, 0] [1, 2] [1, 1] : tensor<2x2xi32> to tensor<1x2xi32>
    %collapsed = tensor.collapse_shape %extracted_slice [[0, 1]] : tensor<1x2xi32> into tensor<2xi32>
    %extracted_slice_7 = tensor.extract_slice %0[1, 0] [1, 2] [1, 1] : tensor<2x2xi32> to tensor<1x2xi32>
    %collapsed_8 = tensor.collapse_shape %extracted_slice_7 [[0, 1]] : tensor<1x2xi32> into tensor<2xi32>
    %1 = call @_randint.detensorized(%collapsed_8, %c0_i64, %c96_i64) : (tensor<2xi32>, i64, i64) -> i64
    %extracted = tensor.extract %arg1[] : tensor<f32>
    %extracted_9 = tensor.extract %arg2[] : tensor<f32>
    %2:5 = scf.for %arg8 = %c0 to %c3 step %c1 iter_args(%arg9 = %arg0, %arg10 = %extracted, %arg11 = %extracted_9, %arg12 = %cst, %arg13 = %cst) -> (tensor<4x8x3xf32>, f32, f32, f64, f64) {
      %from_elements_13 = tensor.from_elements %arg10 : tensor<f32>
      %from_elements_14 = tensor.from_elements %arg11 : tensor<f32>
      %6 = arith.index_cast %arg8 : index to i64
      %7 = tensor.empty() : tensor<4x8x3xf32>
      %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_2 : tensor<f32>) outs(%7 : tensor<4x8x3xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<4x8x3xf32>
      %9:4 = scf.for %arg14 = %c0 to %c32 step %c1 iter_args(%arg15 = %8, %arg16 = %cst, %arg17 = %cst, %arg18 = %cst) -> (tensor<4x8x3xf32>, f64, f64, f64) {
        %25 = arith.index_cast %arg14 : index to i64
        %26 = arith.muli %6, %c32_i64 : i64
        %27 = arith.addi %1, %26 : i64
        %28 = arith.addi %27, %25 : i64
        %29 = func.call @remainder.detensorized(%28, %c96_i64) : (i64, i64) -> i64
        %30 = arith.cmpi slt, %29, %c0_i64 : i64
        %31 = arith.addi %29, %c96_i64 : i64
        %32 = arith.select %30, %31, %29 : i64
        %33 = arith.index_cast %32 : i64 to index
        %34 = arith.maxsi %33, %c0 : index
        %35 = arith.minsi %34, %c95 : index
        %extracted_slice_15 = tensor.extract_slice %arg5[%35, 0] [1, 8] [1, 1] : tensor<96x8xf32> to tensor<1x8xf32>
        %collapsed_16 = tensor.collapse_shape %extracted_slice_15 [[0, 1]] : tensor<1x8xf32> into tensor<8xf32>
        %extracted_slice_17 = tensor.extract_slice %arg6[%35] [1] [1] : tensor<96xf32> to tensor<1xf32>
        %collapsed_18 = tensor.collapse_shape %extracted_slice_17 [] : tensor<1xf32> into tensor<f32>
        %extracted_19 = tensor.extract %collapsed_18[] : tensor<f32>
        %extracted_slice_20 = tensor.extract_slice %arg7[%35] [1] [1] : tensor<96xf32> to tensor<1xf32>
        %collapsed_21 = tensor.collapse_shape %extracted_slice_20 [] : tensor<1xf32> into tensor<f32>
        %extracted_22 = tensor.extract %collapsed_21[] : tensor<f32>
        %36 = func.call @qnode_forward_0(%arg9, %collapsed_16) : (tensor<4x8x3xf32>, tensor<8xf32>) -> tensor<f64>
        %extracted_23 = tensor.extract %36[] : tensor<f64>
        %37 = arith.extf %arg11 : f32 to f64
        %38 = arith.mulf %37, %extracted_23 : f64
        %39 = arith.extf %arg10 : f32 to f64
        %40 = arith.addf %38, %39 : f64
        %41 = arith.extf %extracted_19 : f32 to f64
        %42 = arith.extf %extracted_22 : f32 to f64
        %43 = func.call @softplus.detensorized(%40) : (f64) -> f64
        %44 = arith.mulf %41, %40 : f64
        %45 = arith.subf %43, %44 : f64
        %46 = arith.mulf %42, %45 : f64
        %47:3 = gradient.grad "auto" @_sample_loss(%arg9, %from_elements_13, %from_elements_14, %collapsed_16, %collapsed_18, %collapsed_21) {diffArgIndices = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<4x8x3xf32>, tensor<f32>, tensor<f32>, tensor<8xf32>, tensor<f32>, tensor<f32>) -> (tensor<4x8x3xf32>, tensor<f32>, tensor<f32>)
        %extracted_24 = tensor.extract %47#2[] : tensor<f32>
        %extracted_25 = tensor.extract %47#1[] : tensor<f32>
        %48 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg15, %47#0 : tensor<4x8x3xf32>, tensor<4x8x3xf32>) outs(%7 : tensor<4x8x3xf32>) {
        ^bb0(%in: f32, %in_26: f32, %out: f32):
          %54 = arith.addf %in, %in_26 : f32
          linalg.yield %54 : f32
        } -> tensor<4x8x3xf32>
        %49 = arith.extf %extracted_25 : f32 to f64
        %50 = arith.addf %arg16, %49 : f64
        %51 = arith.extf %extracted_24 : f32 to f64
        %52 = arith.addf %arg17, %51 : f64
        %53 = arith.addf %arg18, %46 : f64
        scf.yield %48, %50, %52, %53 : tensor<4x8x3xf32>, f64, f64, f64
      }
      %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_0 : tensor<f32>) outs(%7 : tensor<4x8x3xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<4x8x3xf32>
      %11 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%9#0, %10 : tensor<4x8x3xf32>, tensor<4x8x3xf32>) outs(%7 : tensor<4x8x3xf32>) {
      ^bb0(%in: f32, %in_15: f32, %out: f32):
        %25 = arith.mulf %in, %in_15 : f32
        linalg.yield %25 : f32
      } -> tensor<4x8x3xf32>
      %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_1 : tensor<f32>) outs(%7 : tensor<4x8x3xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<4x8x3xf32>
      %13 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%12, %11 : tensor<4x8x3xf32>, tensor<4x8x3xf32>) outs(%7 : tensor<4x8x3xf32>) {
      ^bb0(%in: f32, %in_15: f32, %out: f32):
        %25 = arith.mulf %in, %in_15 : f32
        linalg.yield %25 : f32
      } -> tensor<4x8x3xf32>
      %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9, %13 : tensor<4x8x3xf32>, tensor<4x8x3xf32>) outs(%7 : tensor<4x8x3xf32>) {
      ^bb0(%in: f32, %in_15: f32, %out: f32):
        %25 = arith.subf %in, %in_15 : f32
        linalg.yield %25 : f32
      } -> tensor<4x8x3xf32>
      %15 = arith.mulf %9#1, %cst_6 : f64
      %16 = arith.truncf %15 : f64 to f32
      %17 = arith.mulf %16, %cst_3 : f32
      %18 = arith.subf %arg10, %17 : f32
      %19 = arith.mulf %9#2, %cst_6 : f64
      %20 = arith.truncf %19 : f64 to f32
      %21 = arith.mulf %20, %cst_3 : f32
      %22 = arith.subf %arg11, %21 : f32
      %23 = arith.divf %9#3, %cst_4 : f64
      %24 = arith.addf %arg12, %23 : f64
      scf.yield %14, %18, %22, %24, %23 : tensor<4x8x3xf32>, f32, f32, f64, f64
    }
    %from_elements = tensor.from_elements %2#1 : tensor<f32>
    %from_elements_10 = tensor.from_elements %2#2 : tensor<f32>
    %3 = arith.divf %2#3, %cst_5 : f64
    %from_elements_11 = tensor.from_elements %3 : tensor<1xf64>
    %from_elements_12 = tensor.from_elements %2#4 : tensor<1xf64>
    %4 = tensor.empty() : tensor<2xf64>
    %5 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%4 : tensor<2xf64>) {
    ^bb0(%out: f64):
      %6 = linalg.index 0 : index
      %7 = arith.cmpi ult, %6, %c1 : index
      %8 = scf.if %7 -> (f64) {
        %extracted_13 = tensor.extract %from_elements_11[%6] : tensor<1xf64>
        scf.yield %extracted_13 : f64
      } else {
        %9 = arith.subi %6, %c1 : index
        %extracted_13 = tensor.extract %from_elements_12[%9] : tensor<1xf64>
        scf.yield %extracted_13 : f64
      }
      linalg.yield %8 : f64
    } -> tensor<2xf64>
    return %2#0, %from_elements, %from_elements_10, %arg3, %collapsed, %5 : tensor<4x8x3xf32>, tensor<f32>, tensor<f32>, tensor<i32>, tensor<2xi32>, tensor<2xf64>
  }
  func.func private @_threefry_split(%arg0: tensor<2xi32>) -> tensor<2x2xi32> attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c0_i64 = arith.constant 0 : i64
    %c5_i64 = arith.constant 5 : i64
    %c1_i64 = arith.constant 1 : i64
    %cst = arith.constant dense<1> : tensor<i64>
    %cst_0 = arith.constant dense<[17, 29, 16, 24]> : tensor<4xi32>
    %cst_1 = arith.constant dense<[13, 15, 26, 6]> : tensor<4xi32>
    %cst_2 = arith.constant dense<32> : tensor<i64>
    %c64_i64 = arith.constant 64 : i64
    %c466688986_i32 = arith.constant 466688986 : i32
    %c1 = arith.constant 1 : index
    %extracted_slice = tensor.extract_slice %arg0[0] [1] [1] : tensor<2xi32> to tensor<1xi32>
    %collapsed = tensor.collapse_shape %extracted_slice [] : tensor<1xi32> into tensor<i32>
    %extracted = tensor.extract %collapsed[] : tensor<i32>
    %extracted_slice_3 = tensor.extract_slice %arg0[1] [1] [1] : tensor<2xi32> to tensor<1xi32>
    %collapsed_4 = tensor.collapse_shape %extracted_slice_3 [] : tensor<1xi32> into tensor<i32>
    %extracted_5 = tensor.extract %collapsed_4[] : tensor<i32>
    %0 = tensor.empty() : tensor<2xi64>
    %1 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%0 : tensor<2xi64>) {
    ^bb0(%out: i64):
      %18 = linalg.index 0 : index
      %19 = arith.index_cast %18 : index to i64
      linalg.yield %19 : i64
    } -> tensor<2xi64>
    %2 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%cst : tensor<i64>) outs(%0 : tensor<2xi64>) {
    ^bb0(%in: i64, %out: i64):
      linalg.yield %in : i64
    } -> tensor<2xi64>
    %3 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%2, %1 : tensor<2xi64>, tensor<2xi64>) outs(%0 : tensor<2xi64>) {
    ^bb0(%in: i64, %in_9: i64, %out: i64):
      %18 = arith.muli %in, %in_9 : i64
      linalg.yield %18 : i64
    } -> tensor<2xi64>
    %4 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%cst_2 : tensor<i64>) outs(%0 : tensor<2xi64>) {
    ^bb0(%in: i64, %out: i64):
      linalg.yield %in : i64
    } -> tensor<2xi64>
    %5 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%3, %4 : tensor<2xi64>, tensor<2xi64>) outs(%0 : tensor<2xi64>) {
    ^bb0(%in: i64, %in_9: i64, %out: i64):
      %18 = arith.shrui %in, %in_9 : i64
      %19 = arith.cmpi ult, %in_9, %c64_i64 : i64
      %20 = arith.select %19, %18, %c0_i64 : i64
      linalg.yield %20 : i64
    } -> tensor<2xi64>
    %6 = tensor.empty() : tensor<2xi32>
    %7 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%3 : tensor<2xi64>) outs(%6 : tensor<2xi32>) {
    ^bb0(%in: i64, %out: i32):
      %18 = arith.trunci %in : i64 to i32
      linalg.yield %18 : i32
    } -> tensor<2xi32>
    %8 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%5 : tensor<2xi64>) outs(%6 : tensor<2xi32>) {
    ^bb0(%in: i64, %out: i32):
      %18 = arith.trunci %in : i64 to i32
      linalg.yield %18 : i32
    } -> tensor<2xi32>
    %9 = arith.xori %extracted, %extracted_5 : i32
    %10 = arith.xori %9, %c466688986_i32 : i32
    %11 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%collapsed : tensor<i32>) outs(%6 : tensor<2xi32>) {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    } -> tensor<2xi32>
    %12 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%8, %11 : tensor<2xi32>, tensor<2xi32>) outs(%6 : tensor<2xi32>) {
    ^bb0(%in: i32, %in_9: i32, %out: i32):
      %18 = arith.addi %in, %in_9 : i32
      linalg.yield %18 : i32
    } -> tensor<2xi32>
    %13 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%collapsed_4 : tensor<i32>) outs(%6 : tensor<2xi32>) {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    } -> tensor<2xi32>
    %14 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%7, %13 : tensor<2xi32>, tensor<2xi32>) outs(%6 : tensor<2xi32>) {
    ^bb0(%in: i32, %in_9: i32, %out: i32):
      %18 = arith.addi %in, %in_9 : i32
      linalg.yield %18 : i32
    } -> tensor<2xi32>
    %extracted_6 = tensor.extract %collapsed_4[] : tensor<i32>
    %extracted_7 = tensor.extract %collapsed[] : tensor<i32>
    %15:8 = scf.for %arg1 = %c0_i64 to %c5_i64 step %c1_i64 iter_args(%arg2 = %c0_i64, %arg3 = %12, %arg4 = %14, %arg5 = %extracted_6, %arg6 = %10, %arg7 = %extracted_7, %arg8 = %cst_1, %arg9 = %cst_0) -> (i64, tensor<2xi32>, tensor<2xi32>, i32, i32, i32, tensor<4xi32>, tensor<4xi32>)  : i64 {
      %18:8 = func.call @closed_call.detensorized(%arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9) : (i64, tensor<2xi32>, tensor<2xi32>, i32, i32, i32, tensor<4xi32>, tensor<4xi32>) -> (i64, tensor<2xi32>, tensor<2xi32>, i32, i32, i32, tensor<4xi32>, tensor<4xi32>)
      scf.yield %18#0, %18#1, %18#2, %18#3, %18#4, %18#5, %18#6, %18#7 : i64, tensor<2xi32>, tensor<2xi32>, i32, i32, i32, tensor<4xi32>, tensor<4xi32>
    }
    %expanded = tensor.expand_shape %15#1 [[0, 1]] output_shape [2, 1] : tensor<2xi32> into tensor<2x1xi32>
    %expanded_8 = tensor.expand_shape %15#2 [[0, 1]] output_shape [2, 1] : tensor<2xi32> into tensor<2x1xi32>
    %16 = tensor.empty() : tensor<2x2xi32>
    %17 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} outs(%16 : tensor<2x2xi32>) {
    ^bb0(%out: i32):
      %18 = linalg.index 0 : index
      %19 = linalg.index 1 : index
      %20 = arith.cmpi ult, %19, %c1 : index
      %21 = scf.if %20 -> (i32) {
        %extracted_9 = tensor.extract %expanded[%18, %19] : tensor<2x1xi32>
        scf.yield %extracted_9 : i32
      } else {
        %22 = arith.subi %19, %c1 : index
        %extracted_9 = tensor.extract %expanded_8[%18, %22] : tensor<2x1xi32>
        scf.yield %extracted_9 : i32
      }
      linalg.yield %21 : i32
    } -> tensor<2x2xi32>
    return %17 : tensor<2x2xi32>
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
    quantum.device shots(%c0_i64) ["/home/pichau/QAgents/.venv/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.so", "LightningSimulator", "{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}"]
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
  func.func private @_sample_loss(%arg0: tensor<4x8x3xf32>, %arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<8xf32>, %arg4: tensor<f32>, %arg5: tensor<f32>) -> tensor<f64> attributes {llvm.linkage = #llvm.linkage<internal>} {
    %extracted = tensor.extract %arg5[] : tensor<f32>
    %extracted_0 = tensor.extract %arg4[] : tensor<f32>
    %extracted_1 = tensor.extract %arg1[] : tensor<f32>
    %extracted_2 = tensor.extract %arg2[] : tensor<f32>
    %0 = call @qnode_forward_0(%arg0, %arg3) : (tensor<4x8x3xf32>, tensor<8xf32>) -> tensor<f64>
    %extracted_3 = tensor.extract %0[] : tensor<f64>
    %1 = arith.extf %extracted_2 : f32 to f64
    %2 = arith.mulf %1, %extracted_3 : f64
    %3 = arith.extf %extracted_1 : f32 to f64
    %4 = arith.addf %2, %3 : f64
    %5 = arith.extf %extracted_0 : f32 to f64
    %6 = arith.extf %extracted : f32 to f64
    %7 = call @softplus_2.detensorized(%4) : (f64) -> f64
    %8 = arith.mulf %5, %4 : f64
    %9 = arith.subf %7, %8 : f64
    %10 = arith.mulf %6, %9 : f64
    %from_elements = tensor.from_elements %10 : tensor<f64>
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
  func.func private @softplus_2.detensorized(%arg0: f64) -> f64 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %cst = arith.constant 0.000000e+00 : f64
    %0 = arith.maximumf %arg0, %cst : f64
    %1 = arith.cmpf une, %arg0, %arg0 : f64
    %2 = math.absf %arg0 : f64
    %3 = arith.negf %2 : f64
    %4 = math.exp %3 : f64
    %5 = math.log1p %4 : f64
    %6 = arith.addf %0, %5 : f64
    %7 = arith.select %1, %arg0, %6 : f64
    return %7 : f64
  }
  func.func private @_where.detensorized(%arg0: i1, %arg1: i64, %arg2: i64) -> i64 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %0 = arith.select %arg0, %arg1, %arg2 : i64
    return %0 : i64
  }
  func.func private @closed_call_1.detensorized(%arg0: i64, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: tensor<4xi32>, %arg7: tensor<4xi32>) -> (i64, i32, i32, i32, i32, i32, tensor<4xi32>, tensor<4xi32>) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %0 = arith.addi %arg0, %c1_i64 : i64
    %extracted_slice = tensor.extract_slice %arg6[0] [1] [1] : tensor<4xi32> to tensor<1xi32>
    %collapsed = tensor.collapse_shape %extracted_slice [] : tensor<1xi32> into tensor<i32>
    %extracted = tensor.extract %collapsed[] : tensor<i32>
    %1 = arith.addi %arg1, %arg2 : i32
    %2 = arith.shli %arg2, %extracted : i32
    %3 = arith.cmpi ult, %extracted, %c32_i32 : i32
    %4 = arith.select %3, %2, %c0_i32 : i32
    %5 = arith.subi %c32_i32, %extracted : i32
    %6 = arith.shrui %arg2, %5 : i32
    %7 = arith.cmpi ult, %5, %c32_i32 : i32
    %8 = arith.select %7, %6, %c0_i32 : i32
    %9 = arith.ori %4, %8 : i32
    %10 = arith.xori %1, %9 : i32
    %extracted_slice_0 = tensor.extract_slice %arg6[1] [1] [1] : tensor<4xi32> to tensor<1xi32>
    %collapsed_1 = tensor.collapse_shape %extracted_slice_0 [] : tensor<1xi32> into tensor<i32>
    %extracted_2 = tensor.extract %collapsed_1[] : tensor<i32>
    %11 = arith.addi %1, %10 : i32
    %12 = arith.shli %10, %extracted_2 : i32
    %13 = arith.cmpi ult, %extracted_2, %c32_i32 : i32
    %14 = arith.select %13, %12, %c0_i32 : i32
    %15 = arith.subi %c32_i32, %extracted_2 : i32
    %16 = arith.shrui %10, %15 : i32
    %17 = arith.cmpi ult, %15, %c32_i32 : i32
    %18 = arith.select %17, %16, %c0_i32 : i32
    %19 = arith.ori %14, %18 : i32
    %20 = arith.xori %11, %19 : i32
    %extracted_slice_3 = tensor.extract_slice %arg6[2] [1] [1] : tensor<4xi32> to tensor<1xi32>
    %collapsed_4 = tensor.collapse_shape %extracted_slice_3 [] : tensor<1xi32> into tensor<i32>
    %extracted_5 = tensor.extract %collapsed_4[] : tensor<i32>
    %21 = arith.addi %11, %20 : i32
    %22 = arith.shli %20, %extracted_5 : i32
    %23 = arith.cmpi ult, %extracted_5, %c32_i32 : i32
    %24 = arith.select %23, %22, %c0_i32 : i32
    %25 = arith.subi %c32_i32, %extracted_5 : i32
    %26 = arith.shrui %20, %25 : i32
    %27 = arith.cmpi ult, %25, %c32_i32 : i32
    %28 = arith.select %27, %26, %c0_i32 : i32
    %29 = arith.ori %24, %28 : i32
    %30 = arith.xori %21, %29 : i32
    %extracted_slice_6 = tensor.extract_slice %arg6[3] [1] [1] : tensor<4xi32> to tensor<1xi32>
    %collapsed_7 = tensor.collapse_shape %extracted_slice_6 [] : tensor<1xi32> into tensor<i32>
    %extracted_8 = tensor.extract %collapsed_7[] : tensor<i32>
    %31 = arith.addi %21, %30 : i32
    %32 = arith.shli %30, %extracted_8 : i32
    %33 = arith.cmpi ult, %extracted_8, %c32_i32 : i32
    %34 = arith.select %33, %32, %c0_i32 : i32
    %35 = arith.subi %c32_i32, %extracted_8 : i32
    %36 = arith.shrui %30, %35 : i32
    %37 = arith.cmpi ult, %35, %c32_i32 : i32
    %38 = arith.select %37, %36, %c0_i32 : i32
    %39 = arith.ori %34, %38 : i32
    %40 = arith.xori %31, %39 : i32
    %41 = arith.addi %31, %arg3 : i32
    %42 = arith.addi %40, %arg4 : i32
    %43 = arith.trunci %0 : i64 to i32
    %44 = arith.addi %42, %43 : i32
    return %0, %41, %44, %arg4, %arg5, %arg3, %arg7, %arg6 : i64, i32, i32, i32, i32, i32, tensor<4xi32>, tensor<4xi32>
  }
  func.func private @clip_0.detensorized(%arg0: i64, %arg1: i64, %arg2: i64) -> i64 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %0 = arith.maxsi %arg1, %arg0 : i64
    %1 = arith.minsi %arg2, %0 : i64
    return %1 : i64
  }
  func.func private @clip.detensorized(%arg0: i64, %arg1: i64, %arg2: i64) -> i64 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %0 = arith.maxsi %arg1, %arg0 : i64
    %1 = arith.minsi %arg2, %0 : i64
    return %1 : i64
  }
  func.func private @closed_call.detensorized(%arg0: i64, %arg1: tensor<2xi32>, %arg2: tensor<2xi32>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: tensor<4xi32>, %arg7: tensor<4xi32>) -> (i64, tensor<2xi32>, tensor<2xi32>, i32, i32, i32, tensor<4xi32>, tensor<4xi32>) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %from_elements = tensor.from_elements %arg3 : tensor<i32>
    %from_elements_0 = tensor.from_elements %arg4 : tensor<i32>
    %0 = arith.addi %arg0, %c1_i64 : i64
    %extracted_slice = tensor.extract_slice %arg6[0] [1] [1] : tensor<4xi32> to tensor<1xi32>
    %collapsed = tensor.collapse_shape %extracted_slice [] : tensor<1xi32> into tensor<i32>
    %extracted = tensor.extract %collapsed[] : tensor<i32>
    %1 = tensor.empty() : tensor<2xi32>
    %2 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg1, %arg2 : tensor<2xi32>, tensor<2xi32>) outs(%1 : tensor<2xi32>) {
    ^bb0(%in: i32, %in_15: i32, %out: i32):
      %41 = arith.addi %in, %in_15 : i32
      linalg.yield %41 : i32
    } -> tensor<2xi32>
    %3 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%collapsed : tensor<i32>) outs(%1 : tensor<2xi32>) {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    } -> tensor<2xi32>
    %4 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg2, %3 : tensor<2xi32>, tensor<2xi32>) outs(%1 : tensor<2xi32>) {
    ^bb0(%in: i32, %in_15: i32, %out: i32):
      %41 = arith.shli %in, %in_15 : i32
      %42 = arith.cmpi ult, %in_15, %c32_i32 : i32
      %43 = arith.select %42, %41, %c0_i32 : i32
      linalg.yield %43 : i32
    } -> tensor<2xi32>
    %5 = arith.subi %c32_i32, %extracted : i32
    %from_elements_1 = tensor.from_elements %5 : tensor<i32>
    %6 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%from_elements_1 : tensor<i32>) outs(%1 : tensor<2xi32>) {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    } -> tensor<2xi32>
    %7 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg2, %6 : tensor<2xi32>, tensor<2xi32>) outs(%1 : tensor<2xi32>) {
    ^bb0(%in: i32, %in_15: i32, %out: i32):
      %41 = arith.shrui %in, %in_15 : i32
      %42 = arith.cmpi ult, %in_15, %c32_i32 : i32
      %43 = arith.select %42, %41, %c0_i32 : i32
      linalg.yield %43 : i32
    } -> tensor<2xi32>
    %8 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%4, %7 : tensor<2xi32>, tensor<2xi32>) outs(%1 : tensor<2xi32>) {
    ^bb0(%in: i32, %in_15: i32, %out: i32):
      %41 = arith.ori %in, %in_15 : i32
      linalg.yield %41 : i32
    } -> tensor<2xi32>
    %9 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%2, %8 : tensor<2xi32>, tensor<2xi32>) outs(%1 : tensor<2xi32>) {
    ^bb0(%in: i32, %in_15: i32, %out: i32):
      %41 = arith.xori %in, %in_15 : i32
      linalg.yield %41 : i32
    } -> tensor<2xi32>
    %extracted_slice_2 = tensor.extract_slice %arg6[1] [1] [1] : tensor<4xi32> to tensor<1xi32>
    %collapsed_3 = tensor.collapse_shape %extracted_slice_2 [] : tensor<1xi32> into tensor<i32>
    %extracted_4 = tensor.extract %collapsed_3[] : tensor<i32>
    %10 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%2, %9 : tensor<2xi32>, tensor<2xi32>) outs(%1 : tensor<2xi32>) {
    ^bb0(%in: i32, %in_15: i32, %out: i32):
      %41 = arith.addi %in, %in_15 : i32
      linalg.yield %41 : i32
    } -> tensor<2xi32>
    %11 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%collapsed_3 : tensor<i32>) outs(%1 : tensor<2xi32>) {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    } -> tensor<2xi32>
    %12 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%9, %11 : tensor<2xi32>, tensor<2xi32>) outs(%1 : tensor<2xi32>) {
    ^bb0(%in: i32, %in_15: i32, %out: i32):
      %41 = arith.shli %in, %in_15 : i32
      %42 = arith.cmpi ult, %in_15, %c32_i32 : i32
      %43 = arith.select %42, %41, %c0_i32 : i32
      linalg.yield %43 : i32
    } -> tensor<2xi32>
    %13 = arith.subi %c32_i32, %extracted_4 : i32
    %from_elements_5 = tensor.from_elements %13 : tensor<i32>
    %14 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%from_elements_5 : tensor<i32>) outs(%1 : tensor<2xi32>) {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    } -> tensor<2xi32>
    %15 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%9, %14 : tensor<2xi32>, tensor<2xi32>) outs(%1 : tensor<2xi32>) {
    ^bb0(%in: i32, %in_15: i32, %out: i32):
      %41 = arith.shrui %in, %in_15 : i32
      %42 = arith.cmpi ult, %in_15, %c32_i32 : i32
      %43 = arith.select %42, %41, %c0_i32 : i32
      linalg.yield %43 : i32
    } -> tensor<2xi32>
    %16 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%12, %15 : tensor<2xi32>, tensor<2xi32>) outs(%1 : tensor<2xi32>) {
    ^bb0(%in: i32, %in_15: i32, %out: i32):
      %41 = arith.ori %in, %in_15 : i32
      linalg.yield %41 : i32
    } -> tensor<2xi32>
    %17 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%10, %16 : tensor<2xi32>, tensor<2xi32>) outs(%1 : tensor<2xi32>) {
    ^bb0(%in: i32, %in_15: i32, %out: i32):
      %41 = arith.xori %in, %in_15 : i32
      linalg.yield %41 : i32
    } -> tensor<2xi32>
    %extracted_slice_6 = tensor.extract_slice %arg6[2] [1] [1] : tensor<4xi32> to tensor<1xi32>
    %collapsed_7 = tensor.collapse_shape %extracted_slice_6 [] : tensor<1xi32> into tensor<i32>
    %extracted_8 = tensor.extract %collapsed_7[] : tensor<i32>
    %18 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%10, %17 : tensor<2xi32>, tensor<2xi32>) outs(%1 : tensor<2xi32>) {
    ^bb0(%in: i32, %in_15: i32, %out: i32):
      %41 = arith.addi %in, %in_15 : i32
      linalg.yield %41 : i32
    } -> tensor<2xi32>
    %19 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%collapsed_7 : tensor<i32>) outs(%1 : tensor<2xi32>) {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    } -> tensor<2xi32>
    %20 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%17, %19 : tensor<2xi32>, tensor<2xi32>) outs(%1 : tensor<2xi32>) {
    ^bb0(%in: i32, %in_15: i32, %out: i32):
      %41 = arith.shli %in, %in_15 : i32
      %42 = arith.cmpi ult, %in_15, %c32_i32 : i32
      %43 = arith.select %42, %41, %c0_i32 : i32
      linalg.yield %43 : i32
    } -> tensor<2xi32>
    %21 = arith.subi %c32_i32, %extracted_8 : i32
    %from_elements_9 = tensor.from_elements %21 : tensor<i32>
    %22 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%from_elements_9 : tensor<i32>) outs(%1 : tensor<2xi32>) {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    } -> tensor<2xi32>
    %23 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%17, %22 : tensor<2xi32>, tensor<2xi32>) outs(%1 : tensor<2xi32>) {
    ^bb0(%in: i32, %in_15: i32, %out: i32):
      %41 = arith.shrui %in, %in_15 : i32
      %42 = arith.cmpi ult, %in_15, %c32_i32 : i32
      %43 = arith.select %42, %41, %c0_i32 : i32
      linalg.yield %43 : i32
    } -> tensor<2xi32>
    %24 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%20, %23 : tensor<2xi32>, tensor<2xi32>) outs(%1 : tensor<2xi32>) {
    ^bb0(%in: i32, %in_15: i32, %out: i32):
      %41 = arith.ori %in, %in_15 : i32
      linalg.yield %41 : i32
    } -> tensor<2xi32>
    %25 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%18, %24 : tensor<2xi32>, tensor<2xi32>) outs(%1 : tensor<2xi32>) {
    ^bb0(%in: i32, %in_15: i32, %out: i32):
      %41 = arith.xori %in, %in_15 : i32
      linalg.yield %41 : i32
    } -> tensor<2xi32>
    %extracted_slice_10 = tensor.extract_slice %arg6[3] [1] [1] : tensor<4xi32> to tensor<1xi32>
    %collapsed_11 = tensor.collapse_shape %extracted_slice_10 [] : tensor<1xi32> into tensor<i32>
    %extracted_12 = tensor.extract %collapsed_11[] : tensor<i32>
    %26 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%18, %25 : tensor<2xi32>, tensor<2xi32>) outs(%1 : tensor<2xi32>) {
    ^bb0(%in: i32, %in_15: i32, %out: i32):
      %41 = arith.addi %in, %in_15 : i32
      linalg.yield %41 : i32
    } -> tensor<2xi32>
    %27 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%collapsed_11 : tensor<i32>) outs(%1 : tensor<2xi32>) {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    } -> tensor<2xi32>
    %28 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%25, %27 : tensor<2xi32>, tensor<2xi32>) outs(%1 : tensor<2xi32>) {
    ^bb0(%in: i32, %in_15: i32, %out: i32):
      %41 = arith.shli %in, %in_15 : i32
      %42 = arith.cmpi ult, %in_15, %c32_i32 : i32
      %43 = arith.select %42, %41, %c0_i32 : i32
      linalg.yield %43 : i32
    } -> tensor<2xi32>
    %29 = arith.subi %c32_i32, %extracted_12 : i32
    %from_elements_13 = tensor.from_elements %29 : tensor<i32>
    %30 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%from_elements_13 : tensor<i32>) outs(%1 : tensor<2xi32>) {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    } -> tensor<2xi32>
    %31 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%25, %30 : tensor<2xi32>, tensor<2xi32>) outs(%1 : tensor<2xi32>) {
    ^bb0(%in: i32, %in_15: i32, %out: i32):
      %41 = arith.shrui %in, %in_15 : i32
      %42 = arith.cmpi ult, %in_15, %c32_i32 : i32
      %43 = arith.select %42, %41, %c0_i32 : i32
      linalg.yield %43 : i32
    } -> tensor<2xi32>
    %32 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%28, %31 : tensor<2xi32>, tensor<2xi32>) outs(%1 : tensor<2xi32>) {
    ^bb0(%in: i32, %in_15: i32, %out: i32):
      %41 = arith.ori %in, %in_15 : i32
      linalg.yield %41 : i32
    } -> tensor<2xi32>
    %33 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%26, %32 : tensor<2xi32>, tensor<2xi32>) outs(%1 : tensor<2xi32>) {
    ^bb0(%in: i32, %in_15: i32, %out: i32):
      %41 = arith.xori %in, %in_15 : i32
      linalg.yield %41 : i32
    } -> tensor<2xi32>
    %34 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%from_elements : tensor<i32>) outs(%1 : tensor<2xi32>) {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    } -> tensor<2xi32>
    %35 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%26, %34 : tensor<2xi32>, tensor<2xi32>) outs(%1 : tensor<2xi32>) {
    ^bb0(%in: i32, %in_15: i32, %out: i32):
      %41 = arith.addi %in, %in_15 : i32
      linalg.yield %41 : i32
    } -> tensor<2xi32>
    %36 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%from_elements_0 : tensor<i32>) outs(%1 : tensor<2xi32>) {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    } -> tensor<2xi32>
    %37 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%33, %36 : tensor<2xi32>, tensor<2xi32>) outs(%1 : tensor<2xi32>) {
    ^bb0(%in: i32, %in_15: i32, %out: i32):
      %41 = arith.addi %in, %in_15 : i32
      linalg.yield %41 : i32
    } -> tensor<2xi32>
    %38 = arith.trunci %0 : i64 to i32
    %from_elements_14 = tensor.from_elements %38 : tensor<i32>
    %39 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%from_elements_14 : tensor<i32>) outs(%1 : tensor<2xi32>) {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    } -> tensor<2xi32>
    %40 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%37, %39 : tensor<2xi32>, tensor<2xi32>) outs(%1 : tensor<2xi32>) {
    ^bb0(%in: i32, %in_15: i32, %out: i32):
      %41 = arith.addi %in, %in_15 : i32
      linalg.yield %41 : i32
    } -> tensor<2xi32>
    return %0, %35, %40, %arg4, %arg5, %arg3, %arg7, %arg6 : i64, tensor<2xi32>, tensor<2xi32>, i32, i32, i32, tensor<4xi32>, tensor<4xi32>
  }
  func.func private @softplus.detensorized(%arg0: f64) -> f64 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %cst = arith.constant 0.000000e+00 : f64
    %0 = arith.maximumf %arg0, %cst : f64
    %1 = arith.cmpf une, %arg0, %arg0 : f64
    %2 = math.absf %arg0 : f64
    %3 = arith.negf %2 : f64
    %4 = math.exp %3 : f64
    %5 = math.log1p %4 : f64
    %6 = arith.addf %0, %5 : f64
    %7 = arith.select %1, %arg0, %6 : f64
    return %7 : f64
  }
  func.func private @remainder.detensorized(%arg0: i64, %arg1: i64) -> i64 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c-9223372036854775808_i64 = arith.constant -9223372036854775808 : i64
    %c-1_i64 = arith.constant -1 : i64
    %0 = arith.cmpi eq, %arg1, %c0_i64 : i64
    %1 = call @_where.detensorized(%0, %c1_i64, %arg1) : (i1, i64, i64) -> i64
    %2 = arith.cmpi eq, %1, %c0_i64 : i64
    %3 = arith.cmpi eq, %arg0, %c-9223372036854775808_i64 : i64
    %4 = arith.cmpi eq, %1, %c-1_i64 : i64
    %5 = arith.andi %3, %4 : i1
    %6 = arith.ori %2, %5 : i1
    %7 = arith.select %6, %c1_i64, %1 : i64
    %8 = arith.remsi %arg0, %7 : i64
    %9 = arith.select %5, %c0_i64, %8 : i64
    %10 = arith.select %2, %arg0, %9 : i64
    %11 = arith.cmpi ne, %10, %c0_i64 : i64
    %12 = arith.cmpi slt, %10, %c0_i64 : i64
    %13 = arith.cmpi slt, %1, %c0_i64 : i64
    %14 = arith.cmpi ne, %12, %13 : i1
    %15 = arith.andi %14, %11 : i1
    %16 = arith.addi %10, %1 : i64
    %17 = arith.select %15, %16, %10 : i64
    return %17 : i64
  }
  func.func private @_randint.detensorized(%arg0: tensor<2xi32>, %arg1: i64, %arg2: i64) -> i64 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %cst = arith.constant dense<[17, 29, 16, 24]> : tensor<4xi32>
    %c1_i64 = arith.constant 1 : i64
    %c5_i64 = arith.constant 5 : i64
    %c0_i64 = arith.constant 0 : i64
    %cst_0 = arith.constant dense<[13, 15, 26, 6]> : tensor<4xi32>
    %c466688986_i32 = arith.constant 466688986 : i32
    %c32_i64 = arith.constant 32 : i64
    %c4294967296_i64 = arith.constant 4294967296 : i64
    %c9223372036854775807_i64 = arith.constant 9223372036854775807 : i64
    %c-9223372036854775808_i64 = arith.constant -9223372036854775808 : i64
    %0 = call @clip.detensorized(%c9223372036854775807_i64, %c-9223372036854775808_i64, %c9223372036854775807_i64) : (i64, i64, i64) -> i64
    %1 = arith.cmpi sgt, %arg2, %0 : i64
    %2 = call @clip_0.detensorized(%arg1, %c-9223372036854775808_i64, %c9223372036854775807_i64) : (i64, i64, i64) -> i64
    %3 = call @clip_0.detensorized(%arg2, %c-9223372036854775808_i64, %c9223372036854775807_i64) : (i64, i64, i64) -> i64
    %4 = call @_threefry_split(%arg0) : (tensor<2xi32>) -> tensor<2x2xi32>
    %extracted_slice = tensor.extract_slice %4[0, 0] [1, 2] [1, 1] : tensor<2x2xi32> to tensor<1x2xi32>
    %collapsed = tensor.collapse_shape %extracted_slice [[0, 1]] : tensor<1x2xi32> into tensor<2xi32>
    %extracted_slice_1 = tensor.extract_slice %4[1, 0] [1, 2] [1, 1] : tensor<2x2xi32> to tensor<1x2xi32>
    %collapsed_2 = tensor.collapse_shape %extracted_slice_1 [[0, 1]] : tensor<1x2xi32> into tensor<2xi32>
    %extracted_slice_3 = tensor.extract_slice %collapsed[0] [1] [1] : tensor<2xi32> to tensor<1xi32>
    %collapsed_4 = tensor.collapse_shape %extracted_slice_3 [] : tensor<1xi32> into tensor<i32>
    %extracted = tensor.extract %collapsed_4[] : tensor<i32>
    %extracted_slice_5 = tensor.extract_slice %collapsed[1] [1] [1] : tensor<2xi32> to tensor<1xi32>
    %collapsed_6 = tensor.collapse_shape %extracted_slice_5 [] : tensor<1xi32> into tensor<i32>
    %extracted_7 = tensor.extract %collapsed_6[] : tensor<i32>
    %5 = arith.xori %extracted, %extracted_7 : i32
    %6 = arith.xori %5, %c466688986_i32 : i32
    %extracted_8 = tensor.extract %collapsed_4[] : tensor<i32>
    %extracted_9 = tensor.extract %collapsed_6[] : tensor<i32>
    %extracted_10 = tensor.extract %collapsed_6[] : tensor<i32>
    %extracted_11 = tensor.extract %collapsed_4[] : tensor<i32>
    %7:8 = scf.for %arg3 = %c0_i64 to %c5_i64 step %c1_i64 iter_args(%arg4 = %c0_i64, %arg5 = %extracted_8, %arg6 = %extracted_9, %arg7 = %extracted_10, %arg8 = %6, %arg9 = %extracted_11, %arg10 = %cst_0, %arg11 = %cst) -> (i64, i32, i32, i32, i32, i32, tensor<4xi32>, tensor<4xi32>)  : i64 {
      %50:8 = func.call @closed_call_1.detensorized(%arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11) : (i64, i32, i32, i32, i32, i32, tensor<4xi32>, tensor<4xi32>) -> (i64, i32, i32, i32, i32, i32, tensor<4xi32>, tensor<4xi32>)
      scf.yield %50#0, %50#1, %50#2, %50#3, %50#4, %50#5, %50#6, %50#7 : i64, i32, i32, i32, i32, i32, tensor<4xi32>, tensor<4xi32>
    }
    %8 = arith.extui %7#1 : i32 to i64
    %9 = arith.extui %7#2 : i32 to i64
    %10 = arith.shli %8, %c32_i64 : i64
    %11 = arith.ori %10, %9 : i64
    %extracted_slice_12 = tensor.extract_slice %collapsed_2[0] [1] [1] : tensor<2xi32> to tensor<1xi32>
    %collapsed_13 = tensor.collapse_shape %extracted_slice_12 [] : tensor<1xi32> into tensor<i32>
    %extracted_14 = tensor.extract %collapsed_13[] : tensor<i32>
    %extracted_slice_15 = tensor.extract_slice %collapsed_2[1] [1] [1] : tensor<2xi32> to tensor<1xi32>
    %collapsed_16 = tensor.collapse_shape %extracted_slice_15 [] : tensor<1xi32> into tensor<i32>
    %extracted_17 = tensor.extract %collapsed_16[] : tensor<i32>
    %12 = arith.xori %extracted_14, %extracted_17 : i32
    %13 = arith.xori %12, %c466688986_i32 : i32
    %extracted_18 = tensor.extract %collapsed_13[] : tensor<i32>
    %extracted_19 = tensor.extract %collapsed_16[] : tensor<i32>
    %extracted_20 = tensor.extract %collapsed_16[] : tensor<i32>
    %extracted_21 = tensor.extract %collapsed_13[] : tensor<i32>
    %14:8 = scf.for %arg3 = %c0_i64 to %c5_i64 step %c1_i64 iter_args(%arg4 = %c0_i64, %arg5 = %extracted_18, %arg6 = %extracted_19, %arg7 = %extracted_20, %arg8 = %13, %arg9 = %extracted_21, %arg10 = %cst_0, %arg11 = %cst) -> (i64, i32, i32, i32, i32, i32, tensor<4xi32>, tensor<4xi32>)  : i64 {
      %50:8 = func.call @closed_call_1.detensorized(%arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11) : (i64, i32, i32, i32, i32, i32, tensor<4xi32>, tensor<4xi32>) -> (i64, i32, i32, i32, i32, i32, tensor<4xi32>, tensor<4xi32>)
      scf.yield %50#0, %50#1, %50#2, %50#3, %50#4, %50#5, %50#6, %50#7 : i64, i32, i32, i32, i32, i32, tensor<4xi32>, tensor<4xi32>
    }
    %15 = arith.extui %14#1 : i32 to i64
    %16 = arith.extui %14#2 : i32 to i64
    %17 = arith.shli %15, %c32_i64 : i64
    %18 = arith.ori %17, %16 : i64
    %19 = arith.subi %3, %2 : i64
    %20 = arith.cmpi sle, %3, %2 : i64
    %21 = arith.select %20, %c1_i64, %19 : i64
    %22 = arith.cmpi sgt, %3, %2 : i64
    %23 = arith.andi %1, %22 : i1
    %24 = arith.addi %21, %c1_i64 : i64
    %25 = arith.select %23, %24, %21 : i64
    %26 = arith.cmpi eq, %25, %c0_i64 : i64
    %27 = arith.select %26, %c1_i64, %25 : i64
    %28 = arith.remui %c4294967296_i64, %27 : i64
    %29 = arith.select %26, %c4294967296_i64, %28 : i64
    %30 = arith.muli %29, %29 : i64
    %31 = arith.cmpi eq, %25, %c0_i64 : i64
    %32 = arith.select %31, %c1_i64, %25 : i64
    %33 = arith.remui %30, %32 : i64
    %34 = arith.select %31, %30, %33 : i64
    %35 = arith.cmpi eq, %25, %c0_i64 : i64
    %36 = arith.select %35, %c1_i64, %25 : i64
    %37 = arith.remui %11, %36 : i64
    %38 = arith.select %35, %11, %37 : i64
    %39 = arith.muli %38, %34 : i64
    %40 = arith.cmpi eq, %25, %c0_i64 : i64
    %41 = arith.select %40, %c1_i64, %25 : i64
    %42 = arith.remui %18, %41 : i64
    %43 = arith.select %40, %18, %42 : i64
    %44 = arith.addi %39, %43 : i64
    %45 = arith.cmpi eq, %25, %c0_i64 : i64
    %46 = arith.select %45, %c1_i64, %25 : i64
    %47 = arith.remui %44, %46 : i64
    %48 = arith.select %45, %44, %47 : i64
    %49 = arith.addi %2, %48 : i64
    return %49 : i64
  }
}