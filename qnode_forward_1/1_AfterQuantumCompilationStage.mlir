module @qnode_forward {
  func.func public @jit_qnode_forward(%arg0: tensor<2x4x3xf32>, %arg1: tensor<4xf32>) -> tensor<f64> attributes {llvm.emit_c_interface} {
    %0 = call @qnode_forward_0(%arg0, %arg1) : (tensor<2x4x3xf32>, tensor<4xf32>) -> tensor<f64>
    return %0 : tensor<f64>
  }
  func.func public @qnode_forward_0(%arg0: tensor<2x4x3xf32>, %arg1: tensor<4xf32>) -> tensor<f64> attributes {diff_method = "adjoint", llvm.linkage = #llvm.linkage<internal>, qnode} {
    %c0_i64 = arith.constant 0 : i64
    %cst = stablehlo.constant dense<3.14159274> : tensor<f32>
    %0 = stablehlo.slice %arg0 [1:2, 0:1, 2:3] : (tensor<2x4x3xf32>) -> tensor<1x1x1xf32>
    %1 = stablehlo.reshape %0 : (tensor<1x1x1xf32>) -> tensor<f32>
    %2 = stablehlo.slice %arg0 [1:2, 0:1, 1:2] : (tensor<2x4x3xf32>) -> tensor<1x1x1xf32>
    %3 = stablehlo.reshape %2 : (tensor<1x1x1xf32>) -> tensor<f32>
    %4 = stablehlo.slice %arg0 [1:2, 0:1, 0:1] : (tensor<2x4x3xf32>) -> tensor<1x1x1xf32>
    %5 = stablehlo.reshape %4 : (tensor<1x1x1xf32>) -> tensor<f32>
    %6 = stablehlo.slice %arg0 [0:1, 3:4, 2:3] : (tensor<2x4x3xf32>) -> tensor<1x1x1xf32>
    %7 = stablehlo.reshape %6 : (tensor<1x1x1xf32>) -> tensor<f32>
    %8 = stablehlo.slice %arg0 [0:1, 3:4, 1:2] : (tensor<2x4x3xf32>) -> tensor<1x1x1xf32>
    %9 = stablehlo.reshape %8 : (tensor<1x1x1xf32>) -> tensor<f32>
    %10 = stablehlo.slice %arg0 [0:1, 3:4, 0:1] : (tensor<2x4x3xf32>) -> tensor<1x1x1xf32>
    %11 = stablehlo.reshape %10 : (tensor<1x1x1xf32>) -> tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %13 = stablehlo.multiply %12, %arg1 : tensor<4xf32>
    %14 = stablehlo.slice %13 [3:4] : (tensor<4xf32>) -> tensor<1xf32>
    %15 = stablehlo.reshape %14 : (tensor<1xf32>) -> tensor<f32>
    quantum.device shots(%c0_i64) ["/home/pichau/QAgents/.venv/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.so", "LightningSimulator", "{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}"]
    %16 = quantum.alloc( 4) : !quantum.reg
    %17 = quantum.extract %16[ 3] : !quantum.reg -> !quantum.bit
    %18 = stablehlo.convert %15 : (tensor<f32>) -> tensor<f64>
    %extracted = tensor.extract %18[] : tensor<f64>
    %out_qubits = quantum.custom "RY"(%extracted) %17 : !quantum.bit
    %19 = stablehlo.convert %11 : (tensor<f32>) -> tensor<f64>
    %extracted_0 = tensor.extract %19[] : tensor<f64>
    %out_qubits_1 = quantum.custom "RZ"(%extracted_0) %out_qubits : !quantum.bit
    %20 = stablehlo.convert %9 : (tensor<f32>) -> tensor<f64>
    %extracted_2 = tensor.extract %20[] : tensor<f64>
    %out_qubits_3 = quantum.custom "RY"(%extracted_2) %out_qubits_1 : !quantum.bit
    %21 = stablehlo.convert %7 : (tensor<f32>) -> tensor<f64>
    %extracted_4 = tensor.extract %21[] : tensor<f64>
    %out_qubits_5 = quantum.custom "RZ"(%extracted_4) %out_qubits_3 : !quantum.bit
    %22 = stablehlo.slice %arg0 [0:1, 2:3, 2:3] : (tensor<2x4x3xf32>) -> tensor<1x1x1xf32>
    %23 = stablehlo.reshape %22 : (tensor<1x1x1xf32>) -> tensor<f32>
    %24 = stablehlo.slice %arg0 [0:1, 2:3, 1:2] : (tensor<2x4x3xf32>) -> tensor<1x1x1xf32>
    %25 = stablehlo.reshape %24 : (tensor<1x1x1xf32>) -> tensor<f32>
    %26 = stablehlo.slice %arg0 [0:1, 2:3, 0:1] : (tensor<2x4x3xf32>) -> tensor<1x1x1xf32>
    %27 = stablehlo.reshape %26 : (tensor<1x1x1xf32>) -> tensor<f32>
    %28 = stablehlo.slice %13 [2:3] : (tensor<4xf32>) -> tensor<1xf32>
    %29 = stablehlo.reshape %28 : (tensor<1xf32>) -> tensor<f32>
    %30 = quantum.extract %16[ 2] : !quantum.reg -> !quantum.bit
    %31 = stablehlo.convert %29 : (tensor<f32>) -> tensor<f64>
    %extracted_6 = tensor.extract %31[] : tensor<f64>
    %out_qubits_7 = quantum.custom "RY"(%extracted_6) %30 : !quantum.bit
    %32 = stablehlo.convert %27 : (tensor<f32>) -> tensor<f64>
    %extracted_8 = tensor.extract %32[] : tensor<f64>
    %out_qubits_9 = quantum.custom "RZ"(%extracted_8) %out_qubits_7 : !quantum.bit
    %33 = stablehlo.convert %25 : (tensor<f32>) -> tensor<f64>
    %extracted_10 = tensor.extract %33[] : tensor<f64>
    %out_qubits_11 = quantum.custom "RY"(%extracted_10) %out_qubits_9 : !quantum.bit
    %34 = stablehlo.convert %23 : (tensor<f32>) -> tensor<f64>
    %extracted_12 = tensor.extract %34[] : tensor<f64>
    %out_qubits_13 = quantum.custom "RZ"(%extracted_12) %out_qubits_11 : !quantum.bit
    %35 = stablehlo.slice %arg0 [0:1, 0:1, 2:3] : (tensor<2x4x3xf32>) -> tensor<1x1x1xf32>
    %36 = stablehlo.reshape %35 : (tensor<1x1x1xf32>) -> tensor<f32>
    %37 = stablehlo.slice %arg0 [0:1, 0:1, 1:2] : (tensor<2x4x3xf32>) -> tensor<1x1x1xf32>
    %38 = stablehlo.reshape %37 : (tensor<1x1x1xf32>) -> tensor<f32>
    %39 = stablehlo.slice %arg0 [0:1, 0:1, 0:1] : (tensor<2x4x3xf32>) -> tensor<1x1x1xf32>
    %40 = stablehlo.reshape %39 : (tensor<1x1x1xf32>) -> tensor<f32>
    %41 = stablehlo.slice %13 [0:1] : (tensor<4xf32>) -> tensor<1xf32>
    %42 = stablehlo.reshape %41 : (tensor<1xf32>) -> tensor<f32>
    %43 = quantum.extract %16[ 0] : !quantum.reg -> !quantum.bit
    %44 = stablehlo.convert %42 : (tensor<f32>) -> tensor<f64>
    %extracted_14 = tensor.extract %44[] : tensor<f64>
    %out_qubits_15 = quantum.custom "RY"(%extracted_14) %43 : !quantum.bit
    %45 = stablehlo.convert %40 : (tensor<f32>) -> tensor<f64>
    %extracted_16 = tensor.extract %45[] : tensor<f64>
    %out_qubits_17 = quantum.custom "RZ"(%extracted_16) %out_qubits_15 : !quantum.bit
    %46 = stablehlo.convert %38 : (tensor<f32>) -> tensor<f64>
    %extracted_18 = tensor.extract %46[] : tensor<f64>
    %out_qubits_19 = quantum.custom "RY"(%extracted_18) %out_qubits_17 : !quantum.bit
    %47 = stablehlo.convert %36 : (tensor<f32>) -> tensor<f64>
    %extracted_20 = tensor.extract %47[] : tensor<f64>
    %out_qubits_21 = quantum.custom "RZ"(%extracted_20) %out_qubits_19 : !quantum.bit
    %48 = stablehlo.slice %arg0 [0:1, 1:2, 2:3] : (tensor<2x4x3xf32>) -> tensor<1x1x1xf32>
    %49 = stablehlo.reshape %48 : (tensor<1x1x1xf32>) -> tensor<f32>
    %50 = stablehlo.slice %arg0 [0:1, 1:2, 1:2] : (tensor<2x4x3xf32>) -> tensor<1x1x1xf32>
    %51 = stablehlo.reshape %50 : (tensor<1x1x1xf32>) -> tensor<f32>
    %52 = stablehlo.slice %arg0 [0:1, 1:2, 0:1] : (tensor<2x4x3xf32>) -> tensor<1x1x1xf32>
    %53 = stablehlo.reshape %52 : (tensor<1x1x1xf32>) -> tensor<f32>
    %54 = stablehlo.slice %13 [1:2] : (tensor<4xf32>) -> tensor<1xf32>
    %55 = stablehlo.reshape %54 : (tensor<1xf32>) -> tensor<f32>
    %56 = quantum.extract %16[ 1] : !quantum.reg -> !quantum.bit
    %57 = stablehlo.convert %55 : (tensor<f32>) -> tensor<f64>
    %extracted_22 = tensor.extract %57[] : tensor<f64>
    %out_qubits_23 = quantum.custom "RY"(%extracted_22) %56 : !quantum.bit
    %58 = stablehlo.convert %53 : (tensor<f32>) -> tensor<f64>
    %extracted_24 = tensor.extract %58[] : tensor<f64>
    %out_qubits_25 = quantum.custom "RZ"(%extracted_24) %out_qubits_23 : !quantum.bit
    %59 = stablehlo.convert %51 : (tensor<f32>) -> tensor<f64>
    %extracted_26 = tensor.extract %59[] : tensor<f64>
    %out_qubits_27 = quantum.custom "RY"(%extracted_26) %out_qubits_25 : !quantum.bit
    %60 = stablehlo.convert %49 : (tensor<f32>) -> tensor<f64>
    %extracted_28 = tensor.extract %60[] : tensor<f64>
    %out_qubits_29 = quantum.custom "RZ"(%extracted_28) %out_qubits_27 : !quantum.bit
    %out_qubits_30:2 = quantum.custom "CNOT"() %out_qubits_21, %out_qubits_29 : !quantum.bit, !quantum.bit
    %out_qubits_31:2 = quantum.custom "CNOT"() %out_qubits_30#1, %out_qubits_13 : !quantum.bit, !quantum.bit
    %out_qubits_32:2 = quantum.custom "CNOT"() %out_qubits_31#1, %out_qubits_5 : !quantum.bit, !quantum.bit
    %out_qubits_33:2 = quantum.custom "CNOT"() %out_qubits_32#1, %out_qubits_30#0 : !quantum.bit, !quantum.bit
    %61 = stablehlo.convert %5 : (tensor<f32>) -> tensor<f64>
    %extracted_34 = tensor.extract %61[] : tensor<f64>
    %out_qubits_35 = quantum.custom "RZ"(%extracted_34) %out_qubits_33#1 : !quantum.bit
    %62 = stablehlo.convert %3 : (tensor<f32>) -> tensor<f64>
    %extracted_36 = tensor.extract %62[] : tensor<f64>
    %out_qubits_37 = quantum.custom "RY"(%extracted_36) %out_qubits_35 : !quantum.bit
    %63 = stablehlo.convert %1 : (tensor<f32>) -> tensor<f64>
    %extracted_38 = tensor.extract %63[] : tensor<f64>
    %out_qubits_39 = quantum.custom "RZ"(%extracted_38) %out_qubits_37 : !quantum.bit
    %64 = stablehlo.slice %arg0 [1:2, 2:3, 2:3] : (tensor<2x4x3xf32>) -> tensor<1x1x1xf32>
    %65 = stablehlo.reshape %64 : (tensor<1x1x1xf32>) -> tensor<f32>
    %66 = stablehlo.slice %arg0 [1:2, 2:3, 1:2] : (tensor<2x4x3xf32>) -> tensor<1x1x1xf32>
    %67 = stablehlo.reshape %66 : (tensor<1x1x1xf32>) -> tensor<f32>
    %68 = stablehlo.slice %arg0 [1:2, 2:3, 0:1] : (tensor<2x4x3xf32>) -> tensor<1x1x1xf32>
    %69 = stablehlo.reshape %68 : (tensor<1x1x1xf32>) -> tensor<f32>
    %70 = stablehlo.convert %69 : (tensor<f32>) -> tensor<f64>
    %extracted_40 = tensor.extract %70[] : tensor<f64>
    %out_qubits_41 = quantum.custom "RZ"(%extracted_40) %out_qubits_32#0 : !quantum.bit
    %71 = stablehlo.convert %67 : (tensor<f32>) -> tensor<f64>
    %extracted_42 = tensor.extract %71[] : tensor<f64>
    %out_qubits_43 = quantum.custom "RY"(%extracted_42) %out_qubits_41 : !quantum.bit
    %72 = stablehlo.convert %65 : (tensor<f32>) -> tensor<f64>
    %extracted_44 = tensor.extract %72[] : tensor<f64>
    %out_qubits_45 = quantum.custom "RZ"(%extracted_44) %out_qubits_43 : !quantum.bit
    %out_qubits_46:2 = quantum.custom "CNOT"() %out_qubits_39, %out_qubits_45 : !quantum.bit, !quantum.bit
    %out_qubits_47:2 = quantum.custom "CNOT"() %out_qubits_46#1, %out_qubits_46#0 : !quantum.bit, !quantum.bit
    %73 = stablehlo.slice %arg0 [1:2, 1:2, 2:3] : (tensor<2x4x3xf32>) -> tensor<1x1x1xf32>
    %74 = stablehlo.reshape %73 : (tensor<1x1x1xf32>) -> tensor<f32>
    %75 = stablehlo.slice %arg0 [1:2, 1:2, 1:2] : (tensor<2x4x3xf32>) -> tensor<1x1x1xf32>
    %76 = stablehlo.reshape %75 : (tensor<1x1x1xf32>) -> tensor<f32>
    %77 = stablehlo.slice %arg0 [1:2, 1:2, 0:1] : (tensor<2x4x3xf32>) -> tensor<1x1x1xf32>
    %78 = stablehlo.reshape %77 : (tensor<1x1x1xf32>) -> tensor<f32>
    %79 = stablehlo.convert %78 : (tensor<f32>) -> tensor<f64>
    %extracted_48 = tensor.extract %79[] : tensor<f64>
    %out_qubits_49 = quantum.custom "RZ"(%extracted_48) %out_qubits_31#0 : !quantum.bit
    %80 = stablehlo.convert %76 : (tensor<f32>) -> tensor<f64>
    %extracted_50 = tensor.extract %80[] : tensor<f64>
    %out_qubits_51 = quantum.custom "RY"(%extracted_50) %out_qubits_49 : !quantum.bit
    %81 = stablehlo.convert %74 : (tensor<f32>) -> tensor<f64>
    %extracted_52 = tensor.extract %81[] : tensor<f64>
    %out_qubits_53 = quantum.custom "RZ"(%extracted_52) %out_qubits_51 : !quantum.bit
    %82 = stablehlo.slice %arg0 [1:2, 3:4, 2:3] : (tensor<2x4x3xf32>) -> tensor<1x1x1xf32>
    %83 = stablehlo.reshape %82 : (tensor<1x1x1xf32>) -> tensor<f32>
    %84 = stablehlo.slice %arg0 [1:2, 3:4, 1:2] : (tensor<2x4x3xf32>) -> tensor<1x1x1xf32>
    %85 = stablehlo.reshape %84 : (tensor<1x1x1xf32>) -> tensor<f32>
    %86 = stablehlo.slice %arg0 [1:2, 3:4, 0:1] : (tensor<2x4x3xf32>) -> tensor<1x1x1xf32>
    %87 = stablehlo.reshape %86 : (tensor<1x1x1xf32>) -> tensor<f32>
    %88 = stablehlo.convert %87 : (tensor<f32>) -> tensor<f64>
    %extracted_54 = tensor.extract %88[] : tensor<f64>
    %out_qubits_55 = quantum.custom "RZ"(%extracted_54) %out_qubits_33#0 : !quantum.bit
    %89 = stablehlo.convert %85 : (tensor<f32>) -> tensor<f64>
    %extracted_56 = tensor.extract %89[] : tensor<f64>
    %out_qubits_57 = quantum.custom "RY"(%extracted_56) %out_qubits_55 : !quantum.bit
    %90 = stablehlo.convert %83 : (tensor<f32>) -> tensor<f64>
    %extracted_58 = tensor.extract %90[] : tensor<f64>
    %out_qubits_59 = quantum.custom "RZ"(%extracted_58) %out_qubits_57 : !quantum.bit
    %out_qubits_60:2 = quantum.custom "CNOT"() %out_qubits_53, %out_qubits_59 : !quantum.bit, !quantum.bit
    %out_qubits_61:2 = quantum.custom "CNOT"() %out_qubits_60#1, %out_qubits_60#0 : !quantum.bit, !quantum.bit
    %91 = quantum.namedobs %out_qubits_47#1[ PauliZ] : !quantum.obs
    %92 = quantum.expval %91 : f64
    %from_elements = tensor.from_elements %92 : tensor<f64>
    %93 = quantum.insert %16[ 0], %out_qubits_47#1 : !quantum.reg, !quantum.bit
    %94 = quantum.insert %93[ 1], %out_qubits_61#1 : !quantum.reg, !quantum.bit
    %95 = quantum.insert %94[ 2], %out_qubits_47#0 : !quantum.reg, !quantum.bit
    %96 = quantum.insert %95[ 3], %out_qubits_61#0 : !quantum.reg, !quantum.bit
    quantum.dealloc %96 : !quantum.reg
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