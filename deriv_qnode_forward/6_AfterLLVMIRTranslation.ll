; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

@"{}" = internal constant [3 x i8] c"{}\00"
@LightningGPUSimulator = internal constant [22 x i8] c"LightningGPUSimulator\00"
@"/home/pichau/QAgents/.venv/lib/python3.12/site-packages/pennylane_lightning/liblightning_gpu_catalyst.so" = internal constant [105 x i8] c"/home/pichau/QAgents/.venv/lib/python3.12/site-packages/pennylane_lightning/liblightning_gpu_catalyst.so\00"
@enzyme_dupnoneed = linkonce constant i8 0
@enzyme_const = linkonce constant i8 0
@__enzyme_function_like_free = global [2 x ptr] [ptr @_mlir_memref_to_llvm_free, ptr @freename]
@freename = linkonce constant [5 x i8] c"free\00"
@dealloc_indices = linkonce constant [3 x i8] c"-1\00"
@__enzyme_allocation_like = global [4 x ptr] [ptr @_mlir_memref_to_llvm_alloc, ptr null, ptr @dealloc_indices, ptr @_mlir_memref_to_llvm_free]
@__enzyme_register_gradient_qnode_forward_0.quantum = global [3 x ptr] [ptr @qnode_forward_0.quantum, ptr @qnode_forward_0.quantum.augfwd, ptr @qnode_forward_0.quantum.customqgrad]
@__constant_xf32 = private constant float 0x400921FB60000000, align 64

declare void @__catalyst__rt__finalize()

declare void @__catalyst__rt__initialize(ptr)

declare void @__catalyst__rt__device_release()

declare void @__catalyst__rt__qubit_release_array(ptr)

declare double @__catalyst__qis__Expval(i64)

declare i64 @__catalyst__qis__NamedObs(i64, ptr)

declare void @__catalyst__qis__CNOT(ptr, ptr, ptr)

declare void @__catalyst__qis__RZ(double, ptr, ptr)

declare void @__catalyst__qis__RY(double, ptr, ptr)

declare ptr @__catalyst__rt__array_get_element_ptr_1d(ptr, i64)

declare ptr @__catalyst__rt__qubit_allocate_array(i64)

declare void @__catalyst__rt__device_init(ptr, ptr, ptr, i64, i1)

declare void @__catalyst__qis__Gradient(i64, ...)

declare void @__catalyst__rt__toggle_recorder(i1)

declare void @__enzyme_autodiff0(...)

declare void @_mlir_memref_to_llvm_free(ptr)

declare ptr @_mlir_memref_to_llvm_alloc(i64)

define { ptr, ptr, i64, [3 x i64], [3 x i64] } @jit_deriv_qnode_forward(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, ptr %9, ptr %10, i64 %11, i64 %12, i64 %13) {
  %15 = call i64 @qnode_forward_0.pcount(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, ptr %9, ptr %10, i64 %11, i64 %12, i64 %13)
  %16 = call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %17 = ptrtoint ptr %16 to i64
  %18 = add i64 %17, 63
  %19 = urem i64 %18, 64
  %20 = sub i64 %18, %19
  %21 = inttoptr i64 %20 to ptr
  store double 0.000000e+00, ptr %21, align 8, !tbaa !1
  store double 1.000000e+00, ptr %21, align 8, !tbaa !1
  %22 = call ptr @_mlir_memref_to_llvm_alloc(i64 384)
  %23 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } poison, ptr %22, 0
  %24 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %23, ptr %22, 1
  %25 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %24, i64 0, 2
  %26 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %25, i64 4, 3, 0
  %27 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %26, i64 8, 3, 1
  %28 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %27, i64 3, 3, 2
  %29 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %28, i64 24, 4, 0
  %30 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %29, i64 3, 4, 1
  %31 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %30, i64 1, 4, 2
  %32 = call ptr @_mlir_memref_to_llvm_alloc(i64 8)
  call void @llvm.memset.p0.i64(ptr %22, i8 0, i64 384, i1 false)
  call void (...) @__enzyme_autodiff0(ptr @qnode_forward_0.preprocess, ptr @enzyme_const, ptr %0, ptr %1, ptr %22, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, ptr @enzyme_const, ptr %9, ptr @enzyme_const, ptr %10, i64 %11, i64 %12, i64 %13, ptr @enzyme_const, i64 %15, ptr @enzyme_const, ptr %32, ptr @enzyme_dupnoneed, ptr %32, ptr %21, i64 0)
  call void @_mlir_memref_to_llvm_free(ptr %32)
  call void @_mlir_memref_to_llvm_free(ptr %16)
  %33 = ptrtoint ptr %22 to i64
  %34 = icmp eq i64 3735928559, %33
  br i1 %34, label %35, label %46

35:                                               ; preds = %14
  %36 = call ptr @_mlir_memref_to_llvm_alloc(i64 384)
  %37 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } poison, ptr %36, 0
  %38 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %37, ptr %36, 1
  %39 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %38, i64 0, 2
  %40 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %39, i64 4, 3, 0
  %41 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %40, i64 8, 3, 1
  %42 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %41, i64 3, 3, 2
  %43 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %42, i64 24, 4, 0
  %44 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %43, i64 3, 4, 1
  %45 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %44, i64 1, 4, 2
  call void @llvm.memcpy.p0.p0.i64(ptr %36, ptr %22, i64 384, i1 false)
  br label %47

46:                                               ; preds = %14
  br label %47

47:                                               ; preds = %35, %46
  %48 = phi { ptr, ptr, i64, [3 x i64], [3 x i64] } [ %31, %46 ], [ %45, %35 ]
  br label %49

49:                                               ; preds = %47
  ret { ptr, ptr, i64, [3 x i64], [3 x i64] } %48
}

define void @_catalyst_pyface_jit_deriv_qnode_forward(ptr %0, ptr %1) {
  %3 = load { ptr, ptr, ptr }, ptr %1, align 8
  %4 = extractvalue { ptr, ptr, ptr } %3, 0
  %5 = extractvalue { ptr, ptr, ptr } %3, 1
  call void @_catalyst_ciface_jit_deriv_qnode_forward(ptr %0, ptr %4, ptr %5)
  ret void
}

define void @_catalyst_ciface_jit_deriv_qnode_forward(ptr %0, ptr %1, ptr %2) {
  %4 = load { ptr, ptr, i64, [3 x i64], [3 x i64] }, ptr %1, align 8
  %5 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %4, 0
  %6 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %4, 1
  %7 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %4, 2
  %8 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %4, 3, 0
  %9 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %4, 3, 1
  %10 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %4, 3, 2
  %11 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %4, 4, 0
  %12 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %4, 4, 1
  %13 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %4, 4, 2
  %14 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %2, align 8
  %15 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %14, 0
  %16 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %14, 1
  %17 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %14, 2
  %18 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %14, 3, 0
  %19 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %14, 4, 0
  %20 = call { ptr, ptr, i64, [3 x i64], [3 x i64] } @jit_deriv_qnode_forward(ptr %5, ptr %6, i64 %7, i64 %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13, ptr %15, ptr %16, i64 %17, i64 %18, i64 %19)
  store { ptr, ptr, i64, [3 x i64], [3 x i64] } %20, ptr %0, align 8
  ret void
}

define internal { ptr, ptr, i64 } @qnode_forward_0(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, ptr %9, ptr %10, i64 %11, i64 %12, i64 %13) {
  %15 = getelementptr inbounds float, ptr %1, i32 74
  %16 = load float, ptr %15, align 4, !tbaa !4
  %17 = getelementptr inbounds float, ptr %1, i32 73
  %18 = load float, ptr %17, align 4, !tbaa !4
  %19 = getelementptr inbounds float, ptr %1, i32 72
  %20 = load float, ptr %19, align 4, !tbaa !4
  %21 = getelementptr inbounds float, ptr %1, i32 50
  %22 = load float, ptr %21, align 4, !tbaa !4
  %23 = getelementptr inbounds float, ptr %1, i32 49
  %24 = load float, ptr %23, align 4, !tbaa !4
  %25 = getelementptr inbounds float, ptr %1, i32 48
  %26 = load float, ptr %25, align 4, !tbaa !4
  %27 = getelementptr inbounds float, ptr %1, i32 44
  %28 = load float, ptr %27, align 4, !tbaa !4
  %29 = getelementptr inbounds float, ptr %1, i32 43
  %30 = load float, ptr %29, align 4, !tbaa !4
  %31 = getelementptr inbounds float, ptr %1, i32 42
  %32 = load float, ptr %31, align 4, !tbaa !4
  %33 = getelementptr inbounds float, ptr %1, i32 23
  %34 = load float, ptr %33, align 4, !tbaa !4
  %35 = getelementptr inbounds float, ptr %1, i32 22
  %36 = load float, ptr %35, align 4, !tbaa !4
  %37 = getelementptr inbounds float, ptr %1, i32 21
  %38 = load float, ptr %37, align 4, !tbaa !4
  %39 = call ptr @_mlir_memref_to_llvm_alloc(i64 96)
  %40 = ptrtoint ptr %39 to i64
  %41 = add i64 %40, 63
  %42 = urem i64 %41, 64
  %43 = sub i64 %41, %42
  %44 = inttoptr i64 %43 to ptr
  br label %45

45:                                               ; preds = %48, %14
  %46 = phi i64 [ %51, %48 ], [ 0, %14 ]
  %47 = icmp slt i64 %46, 8
  br i1 %47, label %48, label %52

48:                                               ; preds = %45
  %49 = load float, ptr @__constant_xf32, align 4, !tbaa !4
  %50 = getelementptr inbounds float, ptr %44, i64 %46
  store float %49, ptr %50, align 4, !tbaa !4
  %51 = add i64 %46, 1
  br label %45

52:                                               ; preds = %45
  br label %53

53:                                               ; preds = %56, %52
  %54 = phi i64 [ %63, %56 ], [ 0, %52 ]
  %55 = icmp slt i64 %54, 8
  br i1 %55, label %56, label %64

56:                                               ; preds = %53
  %57 = getelementptr inbounds float, ptr %44, i64 %54
  %58 = load float, ptr %57, align 4, !tbaa !4
  %59 = getelementptr inbounds float, ptr %10, i64 %54
  %60 = load float, ptr %59, align 4, !tbaa !4
  %61 = fmul float %58, %60
  %62 = getelementptr inbounds float, ptr %44, i64 %54
  store float %61, ptr %62, align 4, !tbaa !4
  %63 = add i64 %54, 1
  br label %53

64:                                               ; preds = %53
  %65 = getelementptr inbounds float, ptr %44, i32 7
  %66 = load float, ptr %65, align 4, !tbaa !4
  call void @__catalyst__rt__device_init(ptr @"/home/pichau/QAgents/.venv/lib/python3.12/site-packages/pennylane_lightning/liblightning_gpu_catalyst.so", ptr @LightningGPUSimulator, ptr @"{}", i64 0, i1 false)
  %67 = call ptr @__catalyst__rt__qubit_allocate_array(i64 8)
  %68 = call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %67, i64 7)
  %69 = load ptr, ptr %68, align 8
  %70 = fpext float %66 to double
  call void @__catalyst__qis__RY(double %70, ptr %69, ptr null)
  %71 = fpext float %38 to double
  call void @__catalyst__qis__RZ(double %71, ptr %69, ptr null)
  %72 = fpext float %36 to double
  call void @__catalyst__qis__RY(double %72, ptr %69, ptr null)
  %73 = fpext float %34 to double
  call void @__catalyst__qis__RZ(double %73, ptr %69, ptr null)
  %74 = getelementptr inbounds float, ptr %1, i32 20
  %75 = load float, ptr %74, align 4, !tbaa !4
  %76 = getelementptr inbounds float, ptr %1, i32 19
  %77 = load float, ptr %76, align 4, !tbaa !4
  %78 = getelementptr inbounds float, ptr %1, i32 18
  %79 = load float, ptr %78, align 4, !tbaa !4
  %80 = getelementptr inbounds float, ptr %44, i32 6
  %81 = load float, ptr %80, align 4, !tbaa !4
  %82 = call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %67, i64 6)
  %83 = load ptr, ptr %82, align 8
  %84 = fpext float %81 to double
  call void @__catalyst__qis__RY(double %84, ptr %83, ptr null)
  %85 = fpext float %79 to double
  call void @__catalyst__qis__RZ(double %85, ptr %83, ptr null)
  %86 = fpext float %77 to double
  call void @__catalyst__qis__RY(double %86, ptr %83, ptr null)
  %87 = fpext float %75 to double
  call void @__catalyst__qis__RZ(double %87, ptr %83, ptr null)
  %88 = getelementptr inbounds float, ptr %1, i32 17
  %89 = load float, ptr %88, align 4, !tbaa !4
  %90 = getelementptr inbounds float, ptr %1, i32 16
  %91 = load float, ptr %90, align 4, !tbaa !4
  %92 = getelementptr inbounds float, ptr %1, i32 15
  %93 = load float, ptr %92, align 4, !tbaa !4
  %94 = getelementptr inbounds float, ptr %44, i32 5
  %95 = load float, ptr %94, align 4, !tbaa !4
  %96 = call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %67, i64 5)
  %97 = load ptr, ptr %96, align 8
  %98 = fpext float %95 to double
  call void @__catalyst__qis__RY(double %98, ptr %97, ptr null)
  %99 = fpext float %93 to double
  call void @__catalyst__qis__RZ(double %99, ptr %97, ptr null)
  %100 = fpext float %91 to double
  call void @__catalyst__qis__RY(double %100, ptr %97, ptr null)
  %101 = fpext float %89 to double
  call void @__catalyst__qis__RZ(double %101, ptr %97, ptr null)
  %102 = getelementptr inbounds float, ptr %1, i32 14
  %103 = load float, ptr %102, align 4, !tbaa !4
  %104 = getelementptr inbounds float, ptr %1, i32 13
  %105 = load float, ptr %104, align 4, !tbaa !4
  %106 = getelementptr inbounds float, ptr %1, i32 12
  %107 = load float, ptr %106, align 4, !tbaa !4
  %108 = getelementptr inbounds float, ptr %44, i32 4
  %109 = load float, ptr %108, align 4, !tbaa !4
  %110 = call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %67, i64 4)
  %111 = load ptr, ptr %110, align 8
  %112 = fpext float %109 to double
  call void @__catalyst__qis__RY(double %112, ptr %111, ptr null)
  %113 = fpext float %107 to double
  call void @__catalyst__qis__RZ(double %113, ptr %111, ptr null)
  %114 = fpext float %105 to double
  call void @__catalyst__qis__RY(double %114, ptr %111, ptr null)
  %115 = fpext float %103 to double
  call void @__catalyst__qis__RZ(double %115, ptr %111, ptr null)
  %116 = getelementptr inbounds float, ptr %1, i32 11
  %117 = load float, ptr %116, align 4, !tbaa !4
  %118 = getelementptr inbounds float, ptr %1, i32 10
  %119 = load float, ptr %118, align 4, !tbaa !4
  %120 = getelementptr inbounds float, ptr %1, i32 9
  %121 = load float, ptr %120, align 4, !tbaa !4
  %122 = getelementptr inbounds float, ptr %44, i32 3
  %123 = load float, ptr %122, align 4, !tbaa !4
  %124 = call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %67, i64 3)
  %125 = load ptr, ptr %124, align 8
  %126 = fpext float %123 to double
  call void @__catalyst__qis__RY(double %126, ptr %125, ptr null)
  %127 = fpext float %121 to double
  call void @__catalyst__qis__RZ(double %127, ptr %125, ptr null)
  %128 = fpext float %119 to double
  call void @__catalyst__qis__RY(double %128, ptr %125, ptr null)
  %129 = fpext float %117 to double
  call void @__catalyst__qis__RZ(double %129, ptr %125, ptr null)
  %130 = getelementptr inbounds float, ptr %1, i32 8
  %131 = load float, ptr %130, align 4, !tbaa !4
  %132 = getelementptr inbounds float, ptr %1, i32 7
  %133 = load float, ptr %132, align 4, !tbaa !4
  %134 = getelementptr inbounds float, ptr %1, i32 6
  %135 = load float, ptr %134, align 4, !tbaa !4
  %136 = getelementptr inbounds float, ptr %44, i32 2
  %137 = load float, ptr %136, align 4, !tbaa !4
  %138 = call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %67, i64 2)
  %139 = load ptr, ptr %138, align 8
  %140 = fpext float %137 to double
  call void @__catalyst__qis__RY(double %140, ptr %139, ptr null)
  %141 = fpext float %135 to double
  call void @__catalyst__qis__RZ(double %141, ptr %139, ptr null)
  %142 = fpext float %133 to double
  call void @__catalyst__qis__RY(double %142, ptr %139, ptr null)
  %143 = fpext float %131 to double
  call void @__catalyst__qis__RZ(double %143, ptr %139, ptr null)
  %144 = getelementptr inbounds float, ptr %1, i32 2
  %145 = load float, ptr %144, align 4, !tbaa !4
  %146 = getelementptr inbounds float, ptr %1, i32 1
  %147 = load float, ptr %146, align 4, !tbaa !4
  %148 = load float, ptr %1, align 4, !tbaa !4
  %149 = load float, ptr %44, align 4, !tbaa !4
  %150 = call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %67, i64 0)
  %151 = load ptr, ptr %150, align 8
  %152 = fpext float %149 to double
  call void @__catalyst__qis__RY(double %152, ptr %151, ptr null)
  %153 = fpext float %148 to double
  call void @__catalyst__qis__RZ(double %153, ptr %151, ptr null)
  %154 = fpext float %147 to double
  call void @__catalyst__qis__RY(double %154, ptr %151, ptr null)
  %155 = fpext float %145 to double
  call void @__catalyst__qis__RZ(double %155, ptr %151, ptr null)
  %156 = getelementptr inbounds float, ptr %1, i32 5
  %157 = load float, ptr %156, align 4, !tbaa !4
  %158 = getelementptr inbounds float, ptr %1, i32 4
  %159 = load float, ptr %158, align 4, !tbaa !4
  %160 = getelementptr inbounds float, ptr %1, i32 3
  %161 = load float, ptr %160, align 4, !tbaa !4
  %162 = getelementptr inbounds float, ptr %44, i32 1
  %163 = load float, ptr %162, align 4, !tbaa !4
  call void @_mlir_memref_to_llvm_free(ptr %39)
  %164 = call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %67, i64 1)
  %165 = load ptr, ptr %164, align 8
  %166 = fpext float %163 to double
  call void @__catalyst__qis__RY(double %166, ptr %165, ptr null)
  %167 = fpext float %161 to double
  call void @__catalyst__qis__RZ(double %167, ptr %165, ptr null)
  %168 = fpext float %159 to double
  call void @__catalyst__qis__RY(double %168, ptr %165, ptr null)
  %169 = fpext float %157 to double
  call void @__catalyst__qis__RZ(double %169, ptr %165, ptr null)
  call void @__catalyst__qis__CNOT(ptr %151, ptr %165, ptr null)
  call void @__catalyst__qis__CNOT(ptr %165, ptr %139, ptr null)
  call void @__catalyst__qis__CNOT(ptr %139, ptr %125, ptr null)
  call void @__catalyst__qis__CNOT(ptr %125, ptr %111, ptr null)
  call void @__catalyst__qis__CNOT(ptr %111, ptr %97, ptr null)
  call void @__catalyst__qis__CNOT(ptr %97, ptr %83, ptr null)
  call void @__catalyst__qis__CNOT(ptr %83, ptr %69, ptr null)
  %170 = fpext float %32 to double
  call void @__catalyst__qis__RZ(double %170, ptr %83, ptr null)
  %171 = fpext float %30 to double
  call void @__catalyst__qis__RY(double %171, ptr %83, ptr null)
  %172 = fpext float %28 to double
  call void @__catalyst__qis__RZ(double %172, ptr %83, ptr null)
  %173 = getelementptr inbounds float, ptr %1, i32 38
  %174 = load float, ptr %173, align 4, !tbaa !4
  %175 = getelementptr inbounds float, ptr %1, i32 37
  %176 = load float, ptr %175, align 4, !tbaa !4
  %177 = getelementptr inbounds float, ptr %1, i32 36
  %178 = load float, ptr %177, align 4, !tbaa !4
  %179 = fpext float %178 to double
  call void @__catalyst__qis__RZ(double %179, ptr %111, ptr null)
  %180 = fpext float %176 to double
  call void @__catalyst__qis__RY(double %180, ptr %111, ptr null)
  %181 = fpext float %174 to double
  call void @__catalyst__qis__RZ(double %181, ptr %111, ptr null)
  %182 = getelementptr inbounds float, ptr %1, i32 26
  %183 = load float, ptr %182, align 4, !tbaa !4
  %184 = getelementptr inbounds float, ptr %1, i32 25
  %185 = load float, ptr %184, align 4, !tbaa !4
  %186 = getelementptr inbounds float, ptr %1, i32 24
  %187 = load float, ptr %186, align 4, !tbaa !4
  call void @__catalyst__qis__CNOT(ptr %69, ptr %151, ptr null)
  %188 = fpext float %187 to double
  call void @__catalyst__qis__RZ(double %188, ptr %151, ptr null)
  %189 = fpext float %185 to double
  call void @__catalyst__qis__RY(double %189, ptr %151, ptr null)
  %190 = fpext float %183 to double
  call void @__catalyst__qis__RZ(double %190, ptr %151, ptr null)
  %191 = getelementptr inbounds float, ptr %1, i32 32
  %192 = load float, ptr %191, align 4, !tbaa !4
  %193 = getelementptr inbounds float, ptr %1, i32 31
  %194 = load float, ptr %193, align 4, !tbaa !4
  %195 = getelementptr inbounds float, ptr %1, i32 30
  %196 = load float, ptr %195, align 4, !tbaa !4
  %197 = fpext float %196 to double
  call void @__catalyst__qis__RZ(double %197, ptr %139, ptr null)
  %198 = fpext float %194 to double
  call void @__catalyst__qis__RY(double %198, ptr %139, ptr null)
  %199 = fpext float %192 to double
  call void @__catalyst__qis__RZ(double %199, ptr %139, ptr null)
  call void @__catalyst__qis__CNOT(ptr %151, ptr %139, ptr null)
  call void @__catalyst__qis__CNOT(ptr %139, ptr %111, ptr null)
  call void @__catalyst__qis__CNOT(ptr %111, ptr %83, ptr null)
  call void @__catalyst__qis__CNOT(ptr %83, ptr %151, ptr null)
  %200 = fpext float %26 to double
  call void @__catalyst__qis__RZ(double %200, ptr %151, ptr null)
  %201 = fpext float %24 to double
  call void @__catalyst__qis__RY(double %201, ptr %151, ptr null)
  %202 = fpext float %22 to double
  call void @__catalyst__qis__RZ(double %202, ptr %151, ptr null)
  %203 = getelementptr inbounds float, ptr %1, i32 59
  %204 = load float, ptr %203, align 4, !tbaa !4
  %205 = getelementptr inbounds float, ptr %1, i32 58
  %206 = load float, ptr %205, align 4, !tbaa !4
  %207 = getelementptr inbounds float, ptr %1, i32 57
  %208 = load float, ptr %207, align 4, !tbaa !4
  %209 = getelementptr inbounds float, ptr %1, i32 41
  %210 = load float, ptr %209, align 4, !tbaa !4
  %211 = getelementptr inbounds float, ptr %1, i32 40
  %212 = load float, ptr %211, align 4, !tbaa !4
  %213 = getelementptr inbounds float, ptr %1, i32 39
  %214 = load float, ptr %213, align 4, !tbaa !4
  %215 = fpext float %214 to double
  call void @__catalyst__qis__RZ(double %215, ptr %97, ptr null)
  %216 = fpext float %212 to double
  call void @__catalyst__qis__RY(double %216, ptr %97, ptr null)
  %217 = fpext float %210 to double
  call void @__catalyst__qis__RZ(double %217, ptr %97, ptr null)
  %218 = getelementptr inbounds float, ptr %1, i32 29
  %219 = load float, ptr %218, align 4, !tbaa !4
  %220 = getelementptr inbounds float, ptr %1, i32 28
  %221 = load float, ptr %220, align 4, !tbaa !4
  %222 = getelementptr inbounds float, ptr %1, i32 27
  %223 = load float, ptr %222, align 4, !tbaa !4
  %224 = fpext float %223 to double
  call void @__catalyst__qis__RZ(double %224, ptr %165, ptr null)
  %225 = fpext float %221 to double
  call void @__catalyst__qis__RY(double %225, ptr %165, ptr null)
  %226 = fpext float %219 to double
  call void @__catalyst__qis__RZ(double %226, ptr %165, ptr null)
  %227 = getelementptr inbounds float, ptr %1, i32 35
  %228 = load float, ptr %227, align 4, !tbaa !4
  %229 = getelementptr inbounds float, ptr %1, i32 34
  %230 = load float, ptr %229, align 4, !tbaa !4
  %231 = getelementptr inbounds float, ptr %1, i32 33
  %232 = load float, ptr %231, align 4, !tbaa !4
  %233 = fpext float %232 to double
  call void @__catalyst__qis__RZ(double %233, ptr %125, ptr null)
  %234 = fpext float %230 to double
  call void @__catalyst__qis__RY(double %234, ptr %125, ptr null)
  %235 = fpext float %228 to double
  call void @__catalyst__qis__RZ(double %235, ptr %125, ptr null)
  call void @__catalyst__qis__CNOT(ptr %165, ptr %125, ptr null)
  call void @__catalyst__qis__CNOT(ptr %125, ptr %97, ptr null)
  %236 = fpext float %208 to double
  call void @__catalyst__qis__RZ(double %236, ptr %125, ptr null)
  %237 = fpext float %206 to double
  call void @__catalyst__qis__RY(double %237, ptr %125, ptr null)
  %238 = fpext float %204 to double
  call void @__catalyst__qis__RZ(double %238, ptr %125, ptr null)
  call void @__catalyst__qis__CNOT(ptr %151, ptr %125, ptr null)
  %239 = getelementptr inbounds float, ptr %1, i32 56
  %240 = load float, ptr %239, align 4, !tbaa !4
  %241 = getelementptr inbounds float, ptr %1, i32 55
  %242 = load float, ptr %241, align 4, !tbaa !4
  %243 = getelementptr inbounds float, ptr %1, i32 54
  %244 = load float, ptr %243, align 4, !tbaa !4
  %245 = fpext float %244 to double
  call void @__catalyst__qis__RZ(double %245, ptr %139, ptr null)
  %246 = fpext float %242 to double
  call void @__catalyst__qis__RY(double %246, ptr %139, ptr null)
  %247 = fpext float %240 to double
  call void @__catalyst__qis__RZ(double %247, ptr %139, ptr null)
  %248 = getelementptr inbounds float, ptr %1, i32 65
  %249 = load float, ptr %248, align 4, !tbaa !4
  %250 = getelementptr inbounds float, ptr %1, i32 64
  %251 = load float, ptr %250, align 4, !tbaa !4
  %252 = getelementptr inbounds float, ptr %1, i32 63
  %253 = load float, ptr %252, align 4, !tbaa !4
  %254 = getelementptr inbounds float, ptr %1, i32 47
  %255 = load float, ptr %254, align 4, !tbaa !4
  %256 = getelementptr inbounds float, ptr %1, i32 46
  %257 = load float, ptr %256, align 4, !tbaa !4
  %258 = getelementptr inbounds float, ptr %1, i32 45
  %259 = load float, ptr %258, align 4, !tbaa !4
  %260 = fpext float %259 to double
  call void @__catalyst__qis__RZ(double %260, ptr %69, ptr null)
  %261 = fpext float %257 to double
  call void @__catalyst__qis__RY(double %261, ptr %69, ptr null)
  %262 = fpext float %255 to double
  call void @__catalyst__qis__RZ(double %262, ptr %69, ptr null)
  call void @__catalyst__qis__CNOT(ptr %97, ptr %69, ptr null)
  %263 = fpext float %253 to double
  call void @__catalyst__qis__RZ(double %263, ptr %97, ptr null)
  %264 = fpext float %251 to double
  call void @__catalyst__qis__RY(double %264, ptr %97, ptr null)
  %265 = fpext float %249 to double
  call void @__catalyst__qis__RZ(double %265, ptr %97, ptr null)
  call void @__catalyst__qis__CNOT(ptr %139, ptr %97, ptr null)
  call void @__catalyst__qis__CNOT(ptr %97, ptr %151, ptr null)
  %266 = fpext float %20 to double
  call void @__catalyst__qis__RZ(double %266, ptr %151, ptr null)
  %267 = fpext float %18 to double
  call void @__catalyst__qis__RY(double %267, ptr %151, ptr null)
  %268 = fpext float %16 to double
  call void @__catalyst__qis__RZ(double %268, ptr %151, ptr null)
  %269 = getelementptr inbounds float, ptr %1, i32 86
  %270 = load float, ptr %269, align 4, !tbaa !4
  %271 = getelementptr inbounds float, ptr %1, i32 85
  %272 = load float, ptr %271, align 4, !tbaa !4
  %273 = getelementptr inbounds float, ptr %1, i32 84
  %274 = load float, ptr %273, align 4, !tbaa !4
  %275 = getelementptr inbounds float, ptr %1, i32 71
  %276 = load float, ptr %275, align 4, !tbaa !4
  %277 = getelementptr inbounds float, ptr %1, i32 70
  %278 = load float, ptr %277, align 4, !tbaa !4
  %279 = getelementptr inbounds float, ptr %1, i32 69
  %280 = load float, ptr %279, align 4, !tbaa !4
  call void @__catalyst__qis__CNOT(ptr %69, ptr %165, ptr null)
  %281 = fpext float %280 to double
  call void @__catalyst__qis__RZ(double %281, ptr %69, ptr null)
  %282 = fpext float %278 to double
  call void @__catalyst__qis__RY(double %282, ptr %69, ptr null)
  %283 = fpext float %276 to double
  call void @__catalyst__qis__RZ(double %283, ptr %69, ptr null)
  %284 = getelementptr inbounds float, ptr %1, i32 53
  %285 = load float, ptr %284, align 4, !tbaa !4
  %286 = getelementptr inbounds float, ptr %1, i32 52
  %287 = load float, ptr %286, align 4, !tbaa !4
  %288 = getelementptr inbounds float, ptr %1, i32 51
  %289 = load float, ptr %288, align 4, !tbaa !4
  %290 = fpext float %289 to double
  call void @__catalyst__qis__RZ(double %290, ptr %165, ptr null)
  %291 = fpext float %287 to double
  call void @__catalyst__qis__RY(double %291, ptr %165, ptr null)
  %292 = fpext float %285 to double
  call void @__catalyst__qis__RZ(double %292, ptr %165, ptr null)
  %293 = getelementptr inbounds float, ptr %1, i32 62
  %294 = load float, ptr %293, align 4, !tbaa !4
  %295 = getelementptr inbounds float, ptr %1, i32 61
  %296 = load float, ptr %295, align 4, !tbaa !4
  %297 = getelementptr inbounds float, ptr %1, i32 60
  %298 = load float, ptr %297, align 4, !tbaa !4
  %299 = fpext float %298 to double
  call void @__catalyst__qis__RZ(double %299, ptr %111, ptr null)
  %300 = fpext float %296 to double
  call void @__catalyst__qis__RY(double %300, ptr %111, ptr null)
  %301 = fpext float %294 to double
  call void @__catalyst__qis__RZ(double %301, ptr %111, ptr null)
  call void @__catalyst__qis__CNOT(ptr %165, ptr %111, ptr null)
  call void @__catalyst__qis__CNOT(ptr %111, ptr %69, ptr null)
  %302 = fpext float %274 to double
  call void @__catalyst__qis__RZ(double %302, ptr %111, ptr null)
  %303 = fpext float %272 to double
  call void @__catalyst__qis__RY(double %303, ptr %111, ptr null)
  %304 = fpext float %270 to double
  call void @__catalyst__qis__RZ(double %304, ptr %111, ptr null)
  call void @__catalyst__qis__CNOT(ptr %151, ptr %111, ptr null)
  call void @__catalyst__qis__CNOT(ptr %111, ptr %151, ptr null)
  %305 = getelementptr inbounds float, ptr %1, i32 77
  %306 = load float, ptr %305, align 4, !tbaa !4
  %307 = getelementptr inbounds float, ptr %1, i32 76
  %308 = load float, ptr %307, align 4, !tbaa !4
  %309 = getelementptr inbounds float, ptr %1, i32 75
  %310 = load float, ptr %309, align 4, !tbaa !4
  %311 = getelementptr inbounds float, ptr %1, i32 68
  %312 = load float, ptr %311, align 4, !tbaa !4
  %313 = getelementptr inbounds float, ptr %1, i32 67
  %314 = load float, ptr %313, align 4, !tbaa !4
  %315 = getelementptr inbounds float, ptr %1, i32 66
  %316 = load float, ptr %315, align 4, !tbaa !4
  %317 = fpext float %316 to double
  call void @__catalyst__qis__RZ(double %317, ptr %83, ptr null)
  %318 = fpext float %314 to double
  call void @__catalyst__qis__RY(double %318, ptr %83, ptr null)
  %319 = fpext float %312 to double
  call void @__catalyst__qis__RZ(double %319, ptr %83, ptr null)
  call void @__catalyst__qis__CNOT(ptr %125, ptr %83, ptr null)
  call void @__catalyst__qis__CNOT(ptr %83, ptr %165, ptr null)
  %320 = fpext float %310 to double
  call void @__catalyst__qis__RZ(double %320, ptr %165, ptr null)
  %321 = fpext float %308 to double
  call void @__catalyst__qis__RY(double %321, ptr %165, ptr null)
  %322 = fpext float %306 to double
  call void @__catalyst__qis__RZ(double %322, ptr %165, ptr null)
  %323 = getelementptr inbounds float, ptr %1, i32 89
  %324 = load float, ptr %323, align 4, !tbaa !4
  %325 = getelementptr inbounds float, ptr %1, i32 88
  %326 = load float, ptr %325, align 4, !tbaa !4
  %327 = getelementptr inbounds float, ptr %1, i32 87
  %328 = load float, ptr %327, align 4, !tbaa !4
  %329 = fpext float %328 to double
  call void @__catalyst__qis__RZ(double %329, ptr %97, ptr null)
  %330 = fpext float %326 to double
  call void @__catalyst__qis__RY(double %330, ptr %97, ptr null)
  %331 = fpext float %324 to double
  call void @__catalyst__qis__RZ(double %331, ptr %97, ptr null)
  call void @__catalyst__qis__CNOT(ptr %165, ptr %97, ptr null)
  call void @__catalyst__qis__CNOT(ptr %97, ptr %165, ptr null)
  %332 = getelementptr inbounds float, ptr %1, i32 80
  %333 = load float, ptr %332, align 4, !tbaa !4
  %334 = getelementptr inbounds float, ptr %1, i32 79
  %335 = load float, ptr %334, align 4, !tbaa !4
  %336 = getelementptr inbounds float, ptr %1, i32 78
  %337 = load float, ptr %336, align 4, !tbaa !4
  call void @__catalyst__qis__CNOT(ptr %69, ptr %139, ptr null)
  %338 = fpext float %337 to double
  call void @__catalyst__qis__RZ(double %338, ptr %139, ptr null)
  %339 = fpext float %335 to double
  call void @__catalyst__qis__RY(double %339, ptr %139, ptr null)
  %340 = fpext float %333 to double
  call void @__catalyst__qis__RZ(double %340, ptr %139, ptr null)
  %341 = getelementptr inbounds float, ptr %1, i32 92
  %342 = load float, ptr %341, align 4, !tbaa !4
  %343 = getelementptr inbounds float, ptr %1, i32 91
  %344 = load float, ptr %343, align 4, !tbaa !4
  %345 = getelementptr inbounds float, ptr %1, i32 90
  %346 = load float, ptr %345, align 4, !tbaa !4
  %347 = fpext float %346 to double
  call void @__catalyst__qis__RZ(double %347, ptr %83, ptr null)
  %348 = fpext float %344 to double
  call void @__catalyst__qis__RY(double %348, ptr %83, ptr null)
  %349 = fpext float %342 to double
  call void @__catalyst__qis__RZ(double %349, ptr %83, ptr null)
  call void @__catalyst__qis__CNOT(ptr %139, ptr %83, ptr null)
  call void @__catalyst__qis__CNOT(ptr %83, ptr %139, ptr null)
  %350 = getelementptr inbounds float, ptr %1, i32 83
  %351 = load float, ptr %350, align 4, !tbaa !4
  %352 = getelementptr inbounds float, ptr %1, i32 82
  %353 = load float, ptr %352, align 4, !tbaa !4
  %354 = getelementptr inbounds float, ptr %1, i32 81
  %355 = load float, ptr %354, align 4, !tbaa !4
  %356 = fpext float %355 to double
  call void @__catalyst__qis__RZ(double %356, ptr %125, ptr null)
  %357 = fpext float %353 to double
  call void @__catalyst__qis__RY(double %357, ptr %125, ptr null)
  %358 = fpext float %351 to double
  call void @__catalyst__qis__RZ(double %358, ptr %125, ptr null)
  %359 = getelementptr inbounds float, ptr %1, i32 95
  %360 = load float, ptr %359, align 4, !tbaa !4
  %361 = getelementptr inbounds float, ptr %1, i32 94
  %362 = load float, ptr %361, align 4, !tbaa !4
  %363 = getelementptr inbounds float, ptr %1, i32 93
  %364 = load float, ptr %363, align 4, !tbaa !4
  %365 = fpext float %364 to double
  call void @__catalyst__qis__RZ(double %365, ptr %69, ptr null)
  %366 = fpext float %362 to double
  call void @__catalyst__qis__RY(double %366, ptr %69, ptr null)
  %367 = fpext float %360 to double
  call void @__catalyst__qis__RZ(double %367, ptr %69, ptr null)
  call void @__catalyst__qis__CNOT(ptr %125, ptr %69, ptr null)
  call void @__catalyst__qis__CNOT(ptr %69, ptr %125, ptr null)
  %368 = call i64 @__catalyst__qis__NamedObs(i64 3, ptr %151)
  %369 = call double @__catalyst__qis__Expval(i64 %368)
  %370 = call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %371 = ptrtoint ptr %370 to i64
  %372 = add i64 %371, 63
  %373 = urem i64 %372, 64
  %374 = sub i64 %372, %373
  %375 = inttoptr i64 %374 to ptr
  %376 = insertvalue { ptr, ptr, i64 } poison, ptr %370, 0
  %377 = insertvalue { ptr, ptr, i64 } %376, ptr %375, 1
  %378 = insertvalue { ptr, ptr, i64 } %377, i64 0, 2
  store double %369, ptr %375, align 8, !tbaa !1
  call void @__catalyst__rt__qubit_release_array(ptr %67)
  call void @__catalyst__rt__device_release()
  ret { ptr, ptr, i64 } %378
}

define { ptr, ptr, i64, [1 x i64], [1 x i64] } @qnode_forward_0.adjoint(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, ptr %9, ptr %10, i64 %11, i64 %12, i64 %13, i64 %14) {
  %16 = getelementptr double, ptr null, i64 %14
  %17 = ptrtoint ptr %16 to i64
  %18 = call ptr @_mlir_memref_to_llvm_alloc(i64 %17)
  %19 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %18, 0
  %20 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %19, ptr %18, 1
  %21 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, i64 0, 2
  %22 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %21, i64 %14, 3, 0
  %23 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %22, i64 1, 4, 0
  call void @__catalyst__rt__toggle_recorder(i1 true)
  %24 = call { ptr, double } @qnode_forward_0.nodealloc(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, ptr %9, ptr %10, i64 %11, i64 %12, i64 %13)
  %25 = extractvalue { ptr, double } %24, 0
  call void @__catalyst__rt__toggle_recorder(i1 false)
  %26 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %23, ptr %26, align 8
  call void (i64, ...) @__catalyst__qis__Gradient(i64 1, ptr %26)
  call void @__catalyst__rt__qubit_release_array(ptr %25)
  call void @__catalyst__rt__device_release()
  ret { ptr, ptr, i64, [1 x i64], [1 x i64] } %23
}

define { ptr, double } @qnode_forward_0.nodealloc(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, ptr %9, ptr %10, i64 %11, i64 %12, i64 %13) {
  %15 = getelementptr inbounds float, ptr %1, i32 74
  %16 = load float, ptr %15, align 4, !tbaa !4
  %17 = getelementptr inbounds float, ptr %1, i32 73
  %18 = load float, ptr %17, align 4, !tbaa !4
  %19 = getelementptr inbounds float, ptr %1, i32 72
  %20 = load float, ptr %19, align 4, !tbaa !4
  %21 = getelementptr inbounds float, ptr %1, i32 50
  %22 = load float, ptr %21, align 4, !tbaa !4
  %23 = getelementptr inbounds float, ptr %1, i32 49
  %24 = load float, ptr %23, align 4, !tbaa !4
  %25 = getelementptr inbounds float, ptr %1, i32 48
  %26 = load float, ptr %25, align 4, !tbaa !4
  %27 = getelementptr inbounds float, ptr %1, i32 44
  %28 = load float, ptr %27, align 4, !tbaa !4
  %29 = getelementptr inbounds float, ptr %1, i32 43
  %30 = load float, ptr %29, align 4, !tbaa !4
  %31 = getelementptr inbounds float, ptr %1, i32 42
  %32 = load float, ptr %31, align 4, !tbaa !4
  %33 = getelementptr inbounds float, ptr %1, i32 23
  %34 = load float, ptr %33, align 4, !tbaa !4
  %35 = getelementptr inbounds float, ptr %1, i32 22
  %36 = load float, ptr %35, align 4, !tbaa !4
  %37 = getelementptr inbounds float, ptr %1, i32 21
  %38 = load float, ptr %37, align 4, !tbaa !4
  %39 = call ptr @_mlir_memref_to_llvm_alloc(i64 96)
  %40 = ptrtoint ptr %39 to i64
  %41 = add i64 %40, 63
  %42 = urem i64 %41, 64
  %43 = sub i64 %41, %42
  %44 = inttoptr i64 %43 to ptr
  br label %45

45:                                               ; preds = %48, %14
  %46 = phi i64 [ %51, %48 ], [ 0, %14 ]
  %47 = icmp slt i64 %46, 8
  br i1 %47, label %48, label %52

48:                                               ; preds = %45
  %49 = load float, ptr @__constant_xf32, align 4, !tbaa !4
  %50 = getelementptr inbounds float, ptr %44, i64 %46
  store float %49, ptr %50, align 4, !tbaa !4
  %51 = add i64 %46, 1
  br label %45

52:                                               ; preds = %45
  br label %53

53:                                               ; preds = %56, %52
  %54 = phi i64 [ %63, %56 ], [ 0, %52 ]
  %55 = icmp slt i64 %54, 8
  br i1 %55, label %56, label %64

56:                                               ; preds = %53
  %57 = getelementptr inbounds float, ptr %44, i64 %54
  %58 = load float, ptr %57, align 4, !tbaa !4
  %59 = getelementptr inbounds float, ptr %10, i64 %54
  %60 = load float, ptr %59, align 4, !tbaa !4
  %61 = fmul float %58, %60
  %62 = getelementptr inbounds float, ptr %44, i64 %54
  store float %61, ptr %62, align 4, !tbaa !4
  %63 = add i64 %54, 1
  br label %53

64:                                               ; preds = %53
  %65 = getelementptr inbounds float, ptr %44, i32 7
  %66 = load float, ptr %65, align 4, !tbaa !4
  call void @__catalyst__rt__device_init(ptr @"/home/pichau/QAgents/.venv/lib/python3.12/site-packages/pennylane_lightning/liblightning_gpu_catalyst.so", ptr @LightningGPUSimulator, ptr @"{}", i64 0, i1 false)
  %67 = call ptr @__catalyst__rt__qubit_allocate_array(i64 8)
  %68 = call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %67, i64 7)
  %69 = load ptr, ptr %68, align 8
  %70 = fpext float %66 to double
  call void @__catalyst__qis__RY(double %70, ptr %69, ptr null)
  %71 = fpext float %38 to double
  call void @__catalyst__qis__RZ(double %71, ptr %69, ptr null)
  %72 = fpext float %36 to double
  call void @__catalyst__qis__RY(double %72, ptr %69, ptr null)
  %73 = fpext float %34 to double
  call void @__catalyst__qis__RZ(double %73, ptr %69, ptr null)
  %74 = getelementptr inbounds float, ptr %1, i32 20
  %75 = load float, ptr %74, align 4, !tbaa !4
  %76 = getelementptr inbounds float, ptr %1, i32 19
  %77 = load float, ptr %76, align 4, !tbaa !4
  %78 = getelementptr inbounds float, ptr %1, i32 18
  %79 = load float, ptr %78, align 4, !tbaa !4
  %80 = getelementptr inbounds float, ptr %44, i32 6
  %81 = load float, ptr %80, align 4, !tbaa !4
  %82 = call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %67, i64 6)
  %83 = load ptr, ptr %82, align 8
  %84 = fpext float %81 to double
  call void @__catalyst__qis__RY(double %84, ptr %83, ptr null)
  %85 = fpext float %79 to double
  call void @__catalyst__qis__RZ(double %85, ptr %83, ptr null)
  %86 = fpext float %77 to double
  call void @__catalyst__qis__RY(double %86, ptr %83, ptr null)
  %87 = fpext float %75 to double
  call void @__catalyst__qis__RZ(double %87, ptr %83, ptr null)
  %88 = getelementptr inbounds float, ptr %1, i32 17
  %89 = load float, ptr %88, align 4, !tbaa !4
  %90 = getelementptr inbounds float, ptr %1, i32 16
  %91 = load float, ptr %90, align 4, !tbaa !4
  %92 = getelementptr inbounds float, ptr %1, i32 15
  %93 = load float, ptr %92, align 4, !tbaa !4
  %94 = getelementptr inbounds float, ptr %44, i32 5
  %95 = load float, ptr %94, align 4, !tbaa !4
  %96 = call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %67, i64 5)
  %97 = load ptr, ptr %96, align 8
  %98 = fpext float %95 to double
  call void @__catalyst__qis__RY(double %98, ptr %97, ptr null)
  %99 = fpext float %93 to double
  call void @__catalyst__qis__RZ(double %99, ptr %97, ptr null)
  %100 = fpext float %91 to double
  call void @__catalyst__qis__RY(double %100, ptr %97, ptr null)
  %101 = fpext float %89 to double
  call void @__catalyst__qis__RZ(double %101, ptr %97, ptr null)
  %102 = getelementptr inbounds float, ptr %1, i32 14
  %103 = load float, ptr %102, align 4, !tbaa !4
  %104 = getelementptr inbounds float, ptr %1, i32 13
  %105 = load float, ptr %104, align 4, !tbaa !4
  %106 = getelementptr inbounds float, ptr %1, i32 12
  %107 = load float, ptr %106, align 4, !tbaa !4
  %108 = getelementptr inbounds float, ptr %44, i32 4
  %109 = load float, ptr %108, align 4, !tbaa !4
  %110 = call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %67, i64 4)
  %111 = load ptr, ptr %110, align 8
  %112 = fpext float %109 to double
  call void @__catalyst__qis__RY(double %112, ptr %111, ptr null)
  %113 = fpext float %107 to double
  call void @__catalyst__qis__RZ(double %113, ptr %111, ptr null)
  %114 = fpext float %105 to double
  call void @__catalyst__qis__RY(double %114, ptr %111, ptr null)
  %115 = fpext float %103 to double
  call void @__catalyst__qis__RZ(double %115, ptr %111, ptr null)
  %116 = getelementptr inbounds float, ptr %1, i32 11
  %117 = load float, ptr %116, align 4, !tbaa !4
  %118 = getelementptr inbounds float, ptr %1, i32 10
  %119 = load float, ptr %118, align 4, !tbaa !4
  %120 = getelementptr inbounds float, ptr %1, i32 9
  %121 = load float, ptr %120, align 4, !tbaa !4
  %122 = getelementptr inbounds float, ptr %44, i32 3
  %123 = load float, ptr %122, align 4, !tbaa !4
  %124 = call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %67, i64 3)
  %125 = load ptr, ptr %124, align 8
  %126 = fpext float %123 to double
  call void @__catalyst__qis__RY(double %126, ptr %125, ptr null)
  %127 = fpext float %121 to double
  call void @__catalyst__qis__RZ(double %127, ptr %125, ptr null)
  %128 = fpext float %119 to double
  call void @__catalyst__qis__RY(double %128, ptr %125, ptr null)
  %129 = fpext float %117 to double
  call void @__catalyst__qis__RZ(double %129, ptr %125, ptr null)
  %130 = getelementptr inbounds float, ptr %1, i32 8
  %131 = load float, ptr %130, align 4, !tbaa !4
  %132 = getelementptr inbounds float, ptr %1, i32 7
  %133 = load float, ptr %132, align 4, !tbaa !4
  %134 = getelementptr inbounds float, ptr %1, i32 6
  %135 = load float, ptr %134, align 4, !tbaa !4
  %136 = getelementptr inbounds float, ptr %44, i32 2
  %137 = load float, ptr %136, align 4, !tbaa !4
  %138 = call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %67, i64 2)
  %139 = load ptr, ptr %138, align 8
  %140 = fpext float %137 to double
  call void @__catalyst__qis__RY(double %140, ptr %139, ptr null)
  %141 = fpext float %135 to double
  call void @__catalyst__qis__RZ(double %141, ptr %139, ptr null)
  %142 = fpext float %133 to double
  call void @__catalyst__qis__RY(double %142, ptr %139, ptr null)
  %143 = fpext float %131 to double
  call void @__catalyst__qis__RZ(double %143, ptr %139, ptr null)
  %144 = getelementptr inbounds float, ptr %1, i32 2
  %145 = load float, ptr %144, align 4, !tbaa !4
  %146 = getelementptr inbounds float, ptr %1, i32 1
  %147 = load float, ptr %146, align 4, !tbaa !4
  %148 = load float, ptr %1, align 4, !tbaa !4
  %149 = load float, ptr %44, align 4, !tbaa !4
  %150 = call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %67, i64 0)
  %151 = load ptr, ptr %150, align 8
  %152 = fpext float %149 to double
  call void @__catalyst__qis__RY(double %152, ptr %151, ptr null)
  %153 = fpext float %148 to double
  call void @__catalyst__qis__RZ(double %153, ptr %151, ptr null)
  %154 = fpext float %147 to double
  call void @__catalyst__qis__RY(double %154, ptr %151, ptr null)
  %155 = fpext float %145 to double
  call void @__catalyst__qis__RZ(double %155, ptr %151, ptr null)
  %156 = getelementptr inbounds float, ptr %1, i32 5
  %157 = load float, ptr %156, align 4, !tbaa !4
  %158 = getelementptr inbounds float, ptr %1, i32 4
  %159 = load float, ptr %158, align 4, !tbaa !4
  %160 = getelementptr inbounds float, ptr %1, i32 3
  %161 = load float, ptr %160, align 4, !tbaa !4
  %162 = getelementptr inbounds float, ptr %44, i32 1
  %163 = load float, ptr %162, align 4, !tbaa !4
  call void @_mlir_memref_to_llvm_free(ptr %39)
  %164 = call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %67, i64 1)
  %165 = load ptr, ptr %164, align 8
  %166 = fpext float %163 to double
  call void @__catalyst__qis__RY(double %166, ptr %165, ptr null)
  %167 = fpext float %161 to double
  call void @__catalyst__qis__RZ(double %167, ptr %165, ptr null)
  %168 = fpext float %159 to double
  call void @__catalyst__qis__RY(double %168, ptr %165, ptr null)
  %169 = fpext float %157 to double
  call void @__catalyst__qis__RZ(double %169, ptr %165, ptr null)
  call void @__catalyst__qis__CNOT(ptr %151, ptr %165, ptr null)
  call void @__catalyst__qis__CNOT(ptr %165, ptr %139, ptr null)
  call void @__catalyst__qis__CNOT(ptr %139, ptr %125, ptr null)
  call void @__catalyst__qis__CNOT(ptr %125, ptr %111, ptr null)
  call void @__catalyst__qis__CNOT(ptr %111, ptr %97, ptr null)
  call void @__catalyst__qis__CNOT(ptr %97, ptr %83, ptr null)
  call void @__catalyst__qis__CNOT(ptr %83, ptr %69, ptr null)
  %170 = fpext float %32 to double
  call void @__catalyst__qis__RZ(double %170, ptr %83, ptr null)
  %171 = fpext float %30 to double
  call void @__catalyst__qis__RY(double %171, ptr %83, ptr null)
  %172 = fpext float %28 to double
  call void @__catalyst__qis__RZ(double %172, ptr %83, ptr null)
  %173 = getelementptr inbounds float, ptr %1, i32 38
  %174 = load float, ptr %173, align 4, !tbaa !4
  %175 = getelementptr inbounds float, ptr %1, i32 37
  %176 = load float, ptr %175, align 4, !tbaa !4
  %177 = getelementptr inbounds float, ptr %1, i32 36
  %178 = load float, ptr %177, align 4, !tbaa !4
  %179 = fpext float %178 to double
  call void @__catalyst__qis__RZ(double %179, ptr %111, ptr null)
  %180 = fpext float %176 to double
  call void @__catalyst__qis__RY(double %180, ptr %111, ptr null)
  %181 = fpext float %174 to double
  call void @__catalyst__qis__RZ(double %181, ptr %111, ptr null)
  %182 = getelementptr inbounds float, ptr %1, i32 26
  %183 = load float, ptr %182, align 4, !tbaa !4
  %184 = getelementptr inbounds float, ptr %1, i32 25
  %185 = load float, ptr %184, align 4, !tbaa !4
  %186 = getelementptr inbounds float, ptr %1, i32 24
  %187 = load float, ptr %186, align 4, !tbaa !4
  call void @__catalyst__qis__CNOT(ptr %69, ptr %151, ptr null)
  %188 = fpext float %187 to double
  call void @__catalyst__qis__RZ(double %188, ptr %151, ptr null)
  %189 = fpext float %185 to double
  call void @__catalyst__qis__RY(double %189, ptr %151, ptr null)
  %190 = fpext float %183 to double
  call void @__catalyst__qis__RZ(double %190, ptr %151, ptr null)
  %191 = getelementptr inbounds float, ptr %1, i32 32
  %192 = load float, ptr %191, align 4, !tbaa !4
  %193 = getelementptr inbounds float, ptr %1, i32 31
  %194 = load float, ptr %193, align 4, !tbaa !4
  %195 = getelementptr inbounds float, ptr %1, i32 30
  %196 = load float, ptr %195, align 4, !tbaa !4
  %197 = fpext float %196 to double
  call void @__catalyst__qis__RZ(double %197, ptr %139, ptr null)
  %198 = fpext float %194 to double
  call void @__catalyst__qis__RY(double %198, ptr %139, ptr null)
  %199 = fpext float %192 to double
  call void @__catalyst__qis__RZ(double %199, ptr %139, ptr null)
  call void @__catalyst__qis__CNOT(ptr %151, ptr %139, ptr null)
  call void @__catalyst__qis__CNOT(ptr %139, ptr %111, ptr null)
  call void @__catalyst__qis__CNOT(ptr %111, ptr %83, ptr null)
  call void @__catalyst__qis__CNOT(ptr %83, ptr %151, ptr null)
  %200 = fpext float %26 to double
  call void @__catalyst__qis__RZ(double %200, ptr %151, ptr null)
  %201 = fpext float %24 to double
  call void @__catalyst__qis__RY(double %201, ptr %151, ptr null)
  %202 = fpext float %22 to double
  call void @__catalyst__qis__RZ(double %202, ptr %151, ptr null)
  %203 = getelementptr inbounds float, ptr %1, i32 59
  %204 = load float, ptr %203, align 4, !tbaa !4
  %205 = getelementptr inbounds float, ptr %1, i32 58
  %206 = load float, ptr %205, align 4, !tbaa !4
  %207 = getelementptr inbounds float, ptr %1, i32 57
  %208 = load float, ptr %207, align 4, !tbaa !4
  %209 = getelementptr inbounds float, ptr %1, i32 41
  %210 = load float, ptr %209, align 4, !tbaa !4
  %211 = getelementptr inbounds float, ptr %1, i32 40
  %212 = load float, ptr %211, align 4, !tbaa !4
  %213 = getelementptr inbounds float, ptr %1, i32 39
  %214 = load float, ptr %213, align 4, !tbaa !4
  %215 = fpext float %214 to double
  call void @__catalyst__qis__RZ(double %215, ptr %97, ptr null)
  %216 = fpext float %212 to double
  call void @__catalyst__qis__RY(double %216, ptr %97, ptr null)
  %217 = fpext float %210 to double
  call void @__catalyst__qis__RZ(double %217, ptr %97, ptr null)
  %218 = getelementptr inbounds float, ptr %1, i32 29
  %219 = load float, ptr %218, align 4, !tbaa !4
  %220 = getelementptr inbounds float, ptr %1, i32 28
  %221 = load float, ptr %220, align 4, !tbaa !4
  %222 = getelementptr inbounds float, ptr %1, i32 27
  %223 = load float, ptr %222, align 4, !tbaa !4
  %224 = fpext float %223 to double
  call void @__catalyst__qis__RZ(double %224, ptr %165, ptr null)
  %225 = fpext float %221 to double
  call void @__catalyst__qis__RY(double %225, ptr %165, ptr null)
  %226 = fpext float %219 to double
  call void @__catalyst__qis__RZ(double %226, ptr %165, ptr null)
  %227 = getelementptr inbounds float, ptr %1, i32 35
  %228 = load float, ptr %227, align 4, !tbaa !4
  %229 = getelementptr inbounds float, ptr %1, i32 34
  %230 = load float, ptr %229, align 4, !tbaa !4
  %231 = getelementptr inbounds float, ptr %1, i32 33
  %232 = load float, ptr %231, align 4, !tbaa !4
  %233 = fpext float %232 to double
  call void @__catalyst__qis__RZ(double %233, ptr %125, ptr null)
  %234 = fpext float %230 to double
  call void @__catalyst__qis__RY(double %234, ptr %125, ptr null)
  %235 = fpext float %228 to double
  call void @__catalyst__qis__RZ(double %235, ptr %125, ptr null)
  call void @__catalyst__qis__CNOT(ptr %165, ptr %125, ptr null)
  call void @__catalyst__qis__CNOT(ptr %125, ptr %97, ptr null)
  %236 = fpext float %208 to double
  call void @__catalyst__qis__RZ(double %236, ptr %125, ptr null)
  %237 = fpext float %206 to double
  call void @__catalyst__qis__RY(double %237, ptr %125, ptr null)
  %238 = fpext float %204 to double
  call void @__catalyst__qis__RZ(double %238, ptr %125, ptr null)
  call void @__catalyst__qis__CNOT(ptr %151, ptr %125, ptr null)
  %239 = getelementptr inbounds float, ptr %1, i32 56
  %240 = load float, ptr %239, align 4, !tbaa !4
  %241 = getelementptr inbounds float, ptr %1, i32 55
  %242 = load float, ptr %241, align 4, !tbaa !4
  %243 = getelementptr inbounds float, ptr %1, i32 54
  %244 = load float, ptr %243, align 4, !tbaa !4
  %245 = fpext float %244 to double
  call void @__catalyst__qis__RZ(double %245, ptr %139, ptr null)
  %246 = fpext float %242 to double
  call void @__catalyst__qis__RY(double %246, ptr %139, ptr null)
  %247 = fpext float %240 to double
  call void @__catalyst__qis__RZ(double %247, ptr %139, ptr null)
  %248 = getelementptr inbounds float, ptr %1, i32 65
  %249 = load float, ptr %248, align 4, !tbaa !4
  %250 = getelementptr inbounds float, ptr %1, i32 64
  %251 = load float, ptr %250, align 4, !tbaa !4
  %252 = getelementptr inbounds float, ptr %1, i32 63
  %253 = load float, ptr %252, align 4, !tbaa !4
  %254 = getelementptr inbounds float, ptr %1, i32 47
  %255 = load float, ptr %254, align 4, !tbaa !4
  %256 = getelementptr inbounds float, ptr %1, i32 46
  %257 = load float, ptr %256, align 4, !tbaa !4
  %258 = getelementptr inbounds float, ptr %1, i32 45
  %259 = load float, ptr %258, align 4, !tbaa !4
  %260 = fpext float %259 to double
  call void @__catalyst__qis__RZ(double %260, ptr %69, ptr null)
  %261 = fpext float %257 to double
  call void @__catalyst__qis__RY(double %261, ptr %69, ptr null)
  %262 = fpext float %255 to double
  call void @__catalyst__qis__RZ(double %262, ptr %69, ptr null)
  call void @__catalyst__qis__CNOT(ptr %97, ptr %69, ptr null)
  %263 = fpext float %253 to double
  call void @__catalyst__qis__RZ(double %263, ptr %97, ptr null)
  %264 = fpext float %251 to double
  call void @__catalyst__qis__RY(double %264, ptr %97, ptr null)
  %265 = fpext float %249 to double
  call void @__catalyst__qis__RZ(double %265, ptr %97, ptr null)
  call void @__catalyst__qis__CNOT(ptr %139, ptr %97, ptr null)
  call void @__catalyst__qis__CNOT(ptr %97, ptr %151, ptr null)
  %266 = fpext float %20 to double
  call void @__catalyst__qis__RZ(double %266, ptr %151, ptr null)
  %267 = fpext float %18 to double
  call void @__catalyst__qis__RY(double %267, ptr %151, ptr null)
  %268 = fpext float %16 to double
  call void @__catalyst__qis__RZ(double %268, ptr %151, ptr null)
  %269 = getelementptr inbounds float, ptr %1, i32 86
  %270 = load float, ptr %269, align 4, !tbaa !4
  %271 = getelementptr inbounds float, ptr %1, i32 85
  %272 = load float, ptr %271, align 4, !tbaa !4
  %273 = getelementptr inbounds float, ptr %1, i32 84
  %274 = load float, ptr %273, align 4, !tbaa !4
  %275 = getelementptr inbounds float, ptr %1, i32 71
  %276 = load float, ptr %275, align 4, !tbaa !4
  %277 = getelementptr inbounds float, ptr %1, i32 70
  %278 = load float, ptr %277, align 4, !tbaa !4
  %279 = getelementptr inbounds float, ptr %1, i32 69
  %280 = load float, ptr %279, align 4, !tbaa !4
  call void @__catalyst__qis__CNOT(ptr %69, ptr %165, ptr null)
  %281 = fpext float %280 to double
  call void @__catalyst__qis__RZ(double %281, ptr %69, ptr null)
  %282 = fpext float %278 to double
  call void @__catalyst__qis__RY(double %282, ptr %69, ptr null)
  %283 = fpext float %276 to double
  call void @__catalyst__qis__RZ(double %283, ptr %69, ptr null)
  %284 = getelementptr inbounds float, ptr %1, i32 53
  %285 = load float, ptr %284, align 4, !tbaa !4
  %286 = getelementptr inbounds float, ptr %1, i32 52
  %287 = load float, ptr %286, align 4, !tbaa !4
  %288 = getelementptr inbounds float, ptr %1, i32 51
  %289 = load float, ptr %288, align 4, !tbaa !4
  %290 = fpext float %289 to double
  call void @__catalyst__qis__RZ(double %290, ptr %165, ptr null)
  %291 = fpext float %287 to double
  call void @__catalyst__qis__RY(double %291, ptr %165, ptr null)
  %292 = fpext float %285 to double
  call void @__catalyst__qis__RZ(double %292, ptr %165, ptr null)
  %293 = getelementptr inbounds float, ptr %1, i32 62
  %294 = load float, ptr %293, align 4, !tbaa !4
  %295 = getelementptr inbounds float, ptr %1, i32 61
  %296 = load float, ptr %295, align 4, !tbaa !4
  %297 = getelementptr inbounds float, ptr %1, i32 60
  %298 = load float, ptr %297, align 4, !tbaa !4
  %299 = fpext float %298 to double
  call void @__catalyst__qis__RZ(double %299, ptr %111, ptr null)
  %300 = fpext float %296 to double
  call void @__catalyst__qis__RY(double %300, ptr %111, ptr null)
  %301 = fpext float %294 to double
  call void @__catalyst__qis__RZ(double %301, ptr %111, ptr null)
  call void @__catalyst__qis__CNOT(ptr %165, ptr %111, ptr null)
  call void @__catalyst__qis__CNOT(ptr %111, ptr %69, ptr null)
  %302 = fpext float %274 to double
  call void @__catalyst__qis__RZ(double %302, ptr %111, ptr null)
  %303 = fpext float %272 to double
  call void @__catalyst__qis__RY(double %303, ptr %111, ptr null)
  %304 = fpext float %270 to double
  call void @__catalyst__qis__RZ(double %304, ptr %111, ptr null)
  call void @__catalyst__qis__CNOT(ptr %151, ptr %111, ptr null)
  call void @__catalyst__qis__CNOT(ptr %111, ptr %151, ptr null)
  %305 = getelementptr inbounds float, ptr %1, i32 77
  %306 = load float, ptr %305, align 4, !tbaa !4
  %307 = getelementptr inbounds float, ptr %1, i32 76
  %308 = load float, ptr %307, align 4, !tbaa !4
  %309 = getelementptr inbounds float, ptr %1, i32 75
  %310 = load float, ptr %309, align 4, !tbaa !4
  %311 = getelementptr inbounds float, ptr %1, i32 68
  %312 = load float, ptr %311, align 4, !tbaa !4
  %313 = getelementptr inbounds float, ptr %1, i32 67
  %314 = load float, ptr %313, align 4, !tbaa !4
  %315 = getelementptr inbounds float, ptr %1, i32 66
  %316 = load float, ptr %315, align 4, !tbaa !4
  %317 = fpext float %316 to double
  call void @__catalyst__qis__RZ(double %317, ptr %83, ptr null)
  %318 = fpext float %314 to double
  call void @__catalyst__qis__RY(double %318, ptr %83, ptr null)
  %319 = fpext float %312 to double
  call void @__catalyst__qis__RZ(double %319, ptr %83, ptr null)
  call void @__catalyst__qis__CNOT(ptr %125, ptr %83, ptr null)
  call void @__catalyst__qis__CNOT(ptr %83, ptr %165, ptr null)
  %320 = fpext float %310 to double
  call void @__catalyst__qis__RZ(double %320, ptr %165, ptr null)
  %321 = fpext float %308 to double
  call void @__catalyst__qis__RY(double %321, ptr %165, ptr null)
  %322 = fpext float %306 to double
  call void @__catalyst__qis__RZ(double %322, ptr %165, ptr null)
  %323 = getelementptr inbounds float, ptr %1, i32 89
  %324 = load float, ptr %323, align 4, !tbaa !4
  %325 = getelementptr inbounds float, ptr %1, i32 88
  %326 = load float, ptr %325, align 4, !tbaa !4
  %327 = getelementptr inbounds float, ptr %1, i32 87
  %328 = load float, ptr %327, align 4, !tbaa !4
  %329 = fpext float %328 to double
  call void @__catalyst__qis__RZ(double %329, ptr %97, ptr null)
  %330 = fpext float %326 to double
  call void @__catalyst__qis__RY(double %330, ptr %97, ptr null)
  %331 = fpext float %324 to double
  call void @__catalyst__qis__RZ(double %331, ptr %97, ptr null)
  call void @__catalyst__qis__CNOT(ptr %165, ptr %97, ptr null)
  call void @__catalyst__qis__CNOT(ptr %97, ptr %165, ptr null)
  %332 = getelementptr inbounds float, ptr %1, i32 80
  %333 = load float, ptr %332, align 4, !tbaa !4
  %334 = getelementptr inbounds float, ptr %1, i32 79
  %335 = load float, ptr %334, align 4, !tbaa !4
  %336 = getelementptr inbounds float, ptr %1, i32 78
  %337 = load float, ptr %336, align 4, !tbaa !4
  call void @__catalyst__qis__CNOT(ptr %69, ptr %139, ptr null)
  %338 = fpext float %337 to double
  call void @__catalyst__qis__RZ(double %338, ptr %139, ptr null)
  %339 = fpext float %335 to double
  call void @__catalyst__qis__RY(double %339, ptr %139, ptr null)
  %340 = fpext float %333 to double
  call void @__catalyst__qis__RZ(double %340, ptr %139, ptr null)
  %341 = getelementptr inbounds float, ptr %1, i32 92
  %342 = load float, ptr %341, align 4, !tbaa !4
  %343 = getelementptr inbounds float, ptr %1, i32 91
  %344 = load float, ptr %343, align 4, !tbaa !4
  %345 = getelementptr inbounds float, ptr %1, i32 90
  %346 = load float, ptr %345, align 4, !tbaa !4
  %347 = fpext float %346 to double
  call void @__catalyst__qis__RZ(double %347, ptr %83, ptr null)
  %348 = fpext float %344 to double
  call void @__catalyst__qis__RY(double %348, ptr %83, ptr null)
  %349 = fpext float %342 to double
  call void @__catalyst__qis__RZ(double %349, ptr %83, ptr null)
  call void @__catalyst__qis__CNOT(ptr %139, ptr %83, ptr null)
  call void @__catalyst__qis__CNOT(ptr %83, ptr %139, ptr null)
  %350 = getelementptr inbounds float, ptr %1, i32 83
  %351 = load float, ptr %350, align 4, !tbaa !4
  %352 = getelementptr inbounds float, ptr %1, i32 82
  %353 = load float, ptr %352, align 4, !tbaa !4
  %354 = getelementptr inbounds float, ptr %1, i32 81
  %355 = load float, ptr %354, align 4, !tbaa !4
  %356 = fpext float %355 to double
  call void @__catalyst__qis__RZ(double %356, ptr %125, ptr null)
  %357 = fpext float %353 to double
  call void @__catalyst__qis__RY(double %357, ptr %125, ptr null)
  %358 = fpext float %351 to double
  call void @__catalyst__qis__RZ(double %358, ptr %125, ptr null)
  %359 = getelementptr inbounds float, ptr %1, i32 95
  %360 = load float, ptr %359, align 4, !tbaa !4
  %361 = getelementptr inbounds float, ptr %1, i32 94
  %362 = load float, ptr %361, align 4, !tbaa !4
  %363 = getelementptr inbounds float, ptr %1, i32 93
  %364 = load float, ptr %363, align 4, !tbaa !4
  %365 = fpext float %364 to double
  call void @__catalyst__qis__RZ(double %365, ptr %69, ptr null)
  %366 = fpext float %362 to double
  call void @__catalyst__qis__RY(double %366, ptr %69, ptr null)
  %367 = fpext float %360 to double
  call void @__catalyst__qis__RZ(double %367, ptr %69, ptr null)
  call void @__catalyst__qis__CNOT(ptr %125, ptr %69, ptr null)
  call void @__catalyst__qis__CNOT(ptr %69, ptr %125, ptr null)
  %368 = call i64 @__catalyst__qis__NamedObs(i64 3, ptr %151)
  %369 = call double @__catalyst__qis__Expval(i64 %368)
  %370 = insertvalue { ptr, double } poison, ptr %67, 0
  %371 = insertvalue { ptr, double } %370, double %369, 1
  ret { ptr, double } %371
}

define i64 @qnode_forward_0.pcount(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, ptr %9, ptr %10, i64 %11, i64 %12, i64 %13) {
  %15 = alloca i64, i64 1, align 8
  store i64 0, ptr %15, align 4, !tbaa !6
  %16 = load i64, ptr %15, align 4, !tbaa !6
  %17 = add i64 %16, 1
  store i64 %17, ptr %15, align 4, !tbaa !6
  %18 = load i64, ptr %15, align 4, !tbaa !6
  %19 = add i64 %18, 1
  store i64 %19, ptr %15, align 4, !tbaa !6
  %20 = load i64, ptr %15, align 4, !tbaa !6
  %21 = add i64 %20, 1
  store i64 %21, ptr %15, align 4, !tbaa !6
  %22 = load i64, ptr %15, align 4, !tbaa !6
  %23 = add i64 %22, 1
  store i64 %23, ptr %15, align 4, !tbaa !6
  %24 = load i64, ptr %15, align 4, !tbaa !6
  %25 = add i64 %24, 1
  store i64 %25, ptr %15, align 4, !tbaa !6
  %26 = load i64, ptr %15, align 4, !tbaa !6
  %27 = add i64 %26, 1
  store i64 %27, ptr %15, align 4, !tbaa !6
  %28 = load i64, ptr %15, align 4, !tbaa !6
  %29 = add i64 %28, 1
  store i64 %29, ptr %15, align 4, !tbaa !6
  %30 = load i64, ptr %15, align 4, !tbaa !6
  %31 = add i64 %30, 1
  store i64 %31, ptr %15, align 4, !tbaa !6
  %32 = load i64, ptr %15, align 4, !tbaa !6
  %33 = add i64 %32, 1
  store i64 %33, ptr %15, align 4, !tbaa !6
  %34 = load i64, ptr %15, align 4, !tbaa !6
  %35 = add i64 %34, 1
  store i64 %35, ptr %15, align 4, !tbaa !6
  %36 = load i64, ptr %15, align 4, !tbaa !6
  %37 = add i64 %36, 1
  store i64 %37, ptr %15, align 4, !tbaa !6
  %38 = load i64, ptr %15, align 4, !tbaa !6
  %39 = add i64 %38, 1
  store i64 %39, ptr %15, align 4, !tbaa !6
  %40 = load i64, ptr %15, align 4, !tbaa !6
  %41 = add i64 %40, 1
  store i64 %41, ptr %15, align 4, !tbaa !6
  %42 = load i64, ptr %15, align 4, !tbaa !6
  %43 = add i64 %42, 1
  store i64 %43, ptr %15, align 4, !tbaa !6
  %44 = load i64, ptr %15, align 4, !tbaa !6
  %45 = add i64 %44, 1
  store i64 %45, ptr %15, align 4, !tbaa !6
  %46 = load i64, ptr %15, align 4, !tbaa !6
  %47 = add i64 %46, 1
  store i64 %47, ptr %15, align 4, !tbaa !6
  %48 = load i64, ptr %15, align 4, !tbaa !6
  %49 = add i64 %48, 1
  store i64 %49, ptr %15, align 4, !tbaa !6
  %50 = load i64, ptr %15, align 4, !tbaa !6
  %51 = add i64 %50, 1
  store i64 %51, ptr %15, align 4, !tbaa !6
  %52 = load i64, ptr %15, align 4, !tbaa !6
  %53 = add i64 %52, 1
  store i64 %53, ptr %15, align 4, !tbaa !6
  %54 = load i64, ptr %15, align 4, !tbaa !6
  %55 = add i64 %54, 1
  store i64 %55, ptr %15, align 4, !tbaa !6
  %56 = load i64, ptr %15, align 4, !tbaa !6
  %57 = add i64 %56, 1
  store i64 %57, ptr %15, align 4, !tbaa !6
  %58 = load i64, ptr %15, align 4, !tbaa !6
  %59 = add i64 %58, 1
  store i64 %59, ptr %15, align 4, !tbaa !6
  %60 = load i64, ptr %15, align 4, !tbaa !6
  %61 = add i64 %60, 1
  store i64 %61, ptr %15, align 4, !tbaa !6
  %62 = load i64, ptr %15, align 4, !tbaa !6
  %63 = add i64 %62, 1
  store i64 %63, ptr %15, align 4, !tbaa !6
  %64 = load i64, ptr %15, align 4, !tbaa !6
  %65 = add i64 %64, 1
  store i64 %65, ptr %15, align 4, !tbaa !6
  %66 = load i64, ptr %15, align 4, !tbaa !6
  %67 = add i64 %66, 1
  store i64 %67, ptr %15, align 4, !tbaa !6
  %68 = load i64, ptr %15, align 4, !tbaa !6
  %69 = add i64 %68, 1
  store i64 %69, ptr %15, align 4, !tbaa !6
  %70 = load i64, ptr %15, align 4, !tbaa !6
  %71 = add i64 %70, 1
  store i64 %71, ptr %15, align 4, !tbaa !6
  %72 = load i64, ptr %15, align 4, !tbaa !6
  %73 = add i64 %72, 1
  store i64 %73, ptr %15, align 4, !tbaa !6
  %74 = load i64, ptr %15, align 4, !tbaa !6
  %75 = add i64 %74, 1
  store i64 %75, ptr %15, align 4, !tbaa !6
  %76 = load i64, ptr %15, align 4, !tbaa !6
  %77 = add i64 %76, 1
  store i64 %77, ptr %15, align 4, !tbaa !6
  %78 = load i64, ptr %15, align 4, !tbaa !6
  %79 = add i64 %78, 1
  store i64 %79, ptr %15, align 4, !tbaa !6
  %80 = load i64, ptr %15, align 4, !tbaa !6
  %81 = add i64 %80, 1
  store i64 %81, ptr %15, align 4, !tbaa !6
  %82 = load i64, ptr %15, align 4, !tbaa !6
  %83 = add i64 %82, 1
  store i64 %83, ptr %15, align 4, !tbaa !6
  %84 = load i64, ptr %15, align 4, !tbaa !6
  %85 = add i64 %84, 1
  store i64 %85, ptr %15, align 4, !tbaa !6
  %86 = load i64, ptr %15, align 4, !tbaa !6
  %87 = add i64 %86, 1
  store i64 %87, ptr %15, align 4, !tbaa !6
  %88 = load i64, ptr %15, align 4, !tbaa !6
  %89 = add i64 %88, 1
  store i64 %89, ptr %15, align 4, !tbaa !6
  %90 = load i64, ptr %15, align 4, !tbaa !6
  %91 = add i64 %90, 1
  store i64 %91, ptr %15, align 4, !tbaa !6
  %92 = load i64, ptr %15, align 4, !tbaa !6
  %93 = add i64 %92, 1
  store i64 %93, ptr %15, align 4, !tbaa !6
  %94 = load i64, ptr %15, align 4, !tbaa !6
  %95 = add i64 %94, 1
  store i64 %95, ptr %15, align 4, !tbaa !6
  %96 = load i64, ptr %15, align 4, !tbaa !6
  %97 = add i64 %96, 1
  store i64 %97, ptr %15, align 4, !tbaa !6
  %98 = load i64, ptr %15, align 4, !tbaa !6
  %99 = add i64 %98, 1
  store i64 %99, ptr %15, align 4, !tbaa !6
  %100 = load i64, ptr %15, align 4, !tbaa !6
  %101 = add i64 %100, 1
  store i64 %101, ptr %15, align 4, !tbaa !6
  %102 = load i64, ptr %15, align 4, !tbaa !6
  %103 = add i64 %102, 1
  store i64 %103, ptr %15, align 4, !tbaa !6
  %104 = load i64, ptr %15, align 4, !tbaa !6
  %105 = add i64 %104, 1
  store i64 %105, ptr %15, align 4, !tbaa !6
  %106 = load i64, ptr %15, align 4, !tbaa !6
  %107 = add i64 %106, 1
  store i64 %107, ptr %15, align 4, !tbaa !6
  %108 = load i64, ptr %15, align 4, !tbaa !6
  %109 = add i64 %108, 1
  store i64 %109, ptr %15, align 4, !tbaa !6
  %110 = load i64, ptr %15, align 4, !tbaa !6
  %111 = add i64 %110, 1
  store i64 %111, ptr %15, align 4, !tbaa !6
  %112 = load i64, ptr %15, align 4, !tbaa !6
  %113 = add i64 %112, 1
  store i64 %113, ptr %15, align 4, !tbaa !6
  %114 = load i64, ptr %15, align 4, !tbaa !6
  %115 = add i64 %114, 1
  store i64 %115, ptr %15, align 4, !tbaa !6
  %116 = load i64, ptr %15, align 4, !tbaa !6
  %117 = add i64 %116, 1
  store i64 %117, ptr %15, align 4, !tbaa !6
  %118 = load i64, ptr %15, align 4, !tbaa !6
  %119 = add i64 %118, 1
  store i64 %119, ptr %15, align 4, !tbaa !6
  %120 = load i64, ptr %15, align 4, !tbaa !6
  %121 = add i64 %120, 1
  store i64 %121, ptr %15, align 4, !tbaa !6
  %122 = load i64, ptr %15, align 4, !tbaa !6
  %123 = add i64 %122, 1
  store i64 %123, ptr %15, align 4, !tbaa !6
  %124 = load i64, ptr %15, align 4, !tbaa !6
  %125 = add i64 %124, 1
  store i64 %125, ptr %15, align 4, !tbaa !6
  %126 = load i64, ptr %15, align 4, !tbaa !6
  %127 = add i64 %126, 1
  store i64 %127, ptr %15, align 4, !tbaa !6
  %128 = load i64, ptr %15, align 4, !tbaa !6
  %129 = add i64 %128, 1
  store i64 %129, ptr %15, align 4, !tbaa !6
  %130 = load i64, ptr %15, align 4, !tbaa !6
  %131 = add i64 %130, 1
  store i64 %131, ptr %15, align 4, !tbaa !6
  %132 = load i64, ptr %15, align 4, !tbaa !6
  %133 = add i64 %132, 1
  store i64 %133, ptr %15, align 4, !tbaa !6
  %134 = load i64, ptr %15, align 4, !tbaa !6
  %135 = add i64 %134, 1
  store i64 %135, ptr %15, align 4, !tbaa !6
  %136 = load i64, ptr %15, align 4, !tbaa !6
  %137 = add i64 %136, 1
  store i64 %137, ptr %15, align 4, !tbaa !6
  %138 = load i64, ptr %15, align 4, !tbaa !6
  %139 = add i64 %138, 1
  store i64 %139, ptr %15, align 4, !tbaa !6
  %140 = load i64, ptr %15, align 4, !tbaa !6
  %141 = add i64 %140, 1
  store i64 %141, ptr %15, align 4, !tbaa !6
  %142 = load i64, ptr %15, align 4, !tbaa !6
  %143 = add i64 %142, 1
  store i64 %143, ptr %15, align 4, !tbaa !6
  %144 = load i64, ptr %15, align 4, !tbaa !6
  %145 = add i64 %144, 1
  store i64 %145, ptr %15, align 4, !tbaa !6
  %146 = load i64, ptr %15, align 4, !tbaa !6
  %147 = add i64 %146, 1
  store i64 %147, ptr %15, align 4, !tbaa !6
  %148 = load i64, ptr %15, align 4, !tbaa !6
  %149 = add i64 %148, 1
  store i64 %149, ptr %15, align 4, !tbaa !6
  %150 = load i64, ptr %15, align 4, !tbaa !6
  %151 = add i64 %150, 1
  store i64 %151, ptr %15, align 4, !tbaa !6
  %152 = load i64, ptr %15, align 4, !tbaa !6
  %153 = add i64 %152, 1
  store i64 %153, ptr %15, align 4, !tbaa !6
  %154 = load i64, ptr %15, align 4, !tbaa !6
  %155 = add i64 %154, 1
  store i64 %155, ptr %15, align 4, !tbaa !6
  %156 = load i64, ptr %15, align 4, !tbaa !6
  %157 = add i64 %156, 1
  store i64 %157, ptr %15, align 4, !tbaa !6
  %158 = load i64, ptr %15, align 4, !tbaa !6
  %159 = add i64 %158, 1
  store i64 %159, ptr %15, align 4, !tbaa !6
  %160 = load i64, ptr %15, align 4, !tbaa !6
  %161 = add i64 %160, 1
  store i64 %161, ptr %15, align 4, !tbaa !6
  %162 = load i64, ptr %15, align 4, !tbaa !6
  %163 = add i64 %162, 1
  store i64 %163, ptr %15, align 4, !tbaa !6
  %164 = load i64, ptr %15, align 4, !tbaa !6
  %165 = add i64 %164, 1
  store i64 %165, ptr %15, align 4, !tbaa !6
  %166 = load i64, ptr %15, align 4, !tbaa !6
  %167 = add i64 %166, 1
  store i64 %167, ptr %15, align 4, !tbaa !6
  %168 = load i64, ptr %15, align 4, !tbaa !6
  %169 = add i64 %168, 1
  store i64 %169, ptr %15, align 4, !tbaa !6
  %170 = load i64, ptr %15, align 4, !tbaa !6
  %171 = add i64 %170, 1
  store i64 %171, ptr %15, align 4, !tbaa !6
  %172 = load i64, ptr %15, align 4, !tbaa !6
  %173 = add i64 %172, 1
  store i64 %173, ptr %15, align 4, !tbaa !6
  %174 = load i64, ptr %15, align 4, !tbaa !6
  %175 = add i64 %174, 1
  store i64 %175, ptr %15, align 4, !tbaa !6
  %176 = load i64, ptr %15, align 4, !tbaa !6
  %177 = add i64 %176, 1
  store i64 %177, ptr %15, align 4, !tbaa !6
  %178 = load i64, ptr %15, align 4, !tbaa !6
  %179 = add i64 %178, 1
  store i64 %179, ptr %15, align 4, !tbaa !6
  %180 = load i64, ptr %15, align 4, !tbaa !6
  %181 = add i64 %180, 1
  store i64 %181, ptr %15, align 4, !tbaa !6
  %182 = load i64, ptr %15, align 4, !tbaa !6
  %183 = add i64 %182, 1
  store i64 %183, ptr %15, align 4, !tbaa !6
  %184 = load i64, ptr %15, align 4, !tbaa !6
  %185 = add i64 %184, 1
  store i64 %185, ptr %15, align 4, !tbaa !6
  %186 = load i64, ptr %15, align 4, !tbaa !6
  %187 = add i64 %186, 1
  store i64 %187, ptr %15, align 4, !tbaa !6
  %188 = load i64, ptr %15, align 4, !tbaa !6
  %189 = add i64 %188, 1
  store i64 %189, ptr %15, align 4, !tbaa !6
  %190 = load i64, ptr %15, align 4, !tbaa !6
  %191 = add i64 %190, 1
  store i64 %191, ptr %15, align 4, !tbaa !6
  %192 = load i64, ptr %15, align 4, !tbaa !6
  %193 = add i64 %192, 1
  store i64 %193, ptr %15, align 4, !tbaa !6
  %194 = load i64, ptr %15, align 4, !tbaa !6
  %195 = add i64 %194, 1
  store i64 %195, ptr %15, align 4, !tbaa !6
  %196 = load i64, ptr %15, align 4, !tbaa !6
  %197 = add i64 %196, 1
  store i64 %197, ptr %15, align 4, !tbaa !6
  %198 = load i64, ptr %15, align 4, !tbaa !6
  %199 = add i64 %198, 1
  store i64 %199, ptr %15, align 4, !tbaa !6
  %200 = load i64, ptr %15, align 4, !tbaa !6
  %201 = add i64 %200, 1
  store i64 %201, ptr %15, align 4, !tbaa !6
  %202 = load i64, ptr %15, align 4, !tbaa !6
  %203 = add i64 %202, 1
  store i64 %203, ptr %15, align 4, !tbaa !6
  %204 = load i64, ptr %15, align 4, !tbaa !6
  %205 = add i64 %204, 1
  store i64 %205, ptr %15, align 4, !tbaa !6
  %206 = load i64, ptr %15, align 4, !tbaa !6
  %207 = add i64 %206, 1
  store i64 %207, ptr %15, align 4, !tbaa !6
  %208 = load i64, ptr %15, align 4, !tbaa !6
  %209 = add i64 %208, 1
  store i64 %209, ptr %15, align 4, !tbaa !6
  %210 = load i64, ptr %15, align 4, !tbaa !6
  %211 = add i64 %210, 1
  store i64 %211, ptr %15, align 4, !tbaa !6
  %212 = load i64, ptr %15, align 4, !tbaa !6
  %213 = add i64 %212, 1
  store i64 %213, ptr %15, align 4, !tbaa !6
  %214 = load i64, ptr %15, align 4, !tbaa !6
  %215 = add i64 %214, 1
  store i64 %215, ptr %15, align 4, !tbaa !6
  %216 = load i64, ptr %15, align 4, !tbaa !6
  %217 = add i64 %216, 1
  store i64 %217, ptr %15, align 4, !tbaa !6
  %218 = load i64, ptr %15, align 4, !tbaa !6
  %219 = add i64 %218, 1
  store i64 %219, ptr %15, align 4, !tbaa !6
  %220 = load i64, ptr %15, align 4, !tbaa !6
  %221 = add i64 %220, 1
  store i64 %221, ptr %15, align 4, !tbaa !6
  %222 = load i64, ptr %15, align 4, !tbaa !6
  %223 = add i64 %222, 1
  store i64 %223, ptr %15, align 4, !tbaa !6
  %224 = load i64, ptr %15, align 4, !tbaa !6
  ret i64 %224
}

define void @qnode_forward_0.quantum.customqgrad(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8) {
  %10 = load { ptr, ptr, i64, [3 x i64], [3 x i64] }, ptr %0, align 8
  %11 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %2, align 8
  %12 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %5, align 8
  %13 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %12, 3
  %14 = alloca [1 x i64], i64 1, align 8
  store [1 x i64] %13, ptr %14, align 4
  %15 = getelementptr inbounds [1 x i64], ptr %14, i32 0, i32 0
  %16 = load i64, ptr %15, align 4
  %17 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %10, 0
  %18 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %10, 1
  %19 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %10, 2
  %20 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %10, 3, 0
  %21 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %10, 3, 1
  %22 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %10, 3, 2
  %23 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %10, 4, 0
  %24 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %10, 4, 1
  %25 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %10, 4, 2
  %26 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %11, 0
  %27 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %11, 1
  %28 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %11, 2
  %29 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %11, 3, 0
  %30 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %11, 4, 0
  %31 = call { ptr, ptr, i64, [1 x i64], [1 x i64] } @qnode_forward_0.adjoint(ptr %17, ptr %18, i64 %19, i64 %20, i64 %21, i64 %22, i64 %23, i64 %24, i64 %25, ptr %26, ptr %27, i64 %28, i64 %29, i64 %30, i64 %16)
  %32 = load { ptr, ptr, i64 }, ptr %7, align 8
  %33 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %31, 3
  %34 = alloca [1 x i64], i64 1, align 8
  store [1 x i64] %33, ptr %34, align 4
  %35 = getelementptr inbounds [1 x i64], ptr %34, i32 0, i32 0
  %36 = load i64, ptr %35, align 4
  br label %37

37:                                               ; preds = %40, %9
  %38 = phi i64 [ %53, %40 ], [ 0, %9 ]
  %39 = icmp slt i64 %38, %36
  br i1 %39, label %40, label %54

40:                                               ; preds = %37
  %41 = extractvalue { ptr, ptr, i64 } %32, 1
  %42 = load double, ptr %41, align 8, !tbaa !1
  %43 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %31, 1
  %44 = getelementptr inbounds double, ptr %43, i64 %38
  %45 = load double, ptr %44, align 8, !tbaa !1
  %46 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %12, 1
  %47 = getelementptr inbounds double, ptr %46, i64 %38
  %48 = load double, ptr %47, align 8, !tbaa !1
  %49 = fmul double %42, %45
  %50 = fadd double %48, %49
  %51 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %12, 1
  %52 = getelementptr inbounds double, ptr %51, i64 %38
  store double %50, ptr %52, align 8, !tbaa !1
  %53 = add i64 %38, 1
  br label %37

54:                                               ; preds = %37
  ret void
}

; Function Attrs: noinline
define void @qnode_forward_0.quantum(ptr %0, ptr %1, ptr %2, ptr %3) #0 {
  %5 = load volatile { ptr, ptr, i64, [3 x i64], [3 x i64] }, ptr %0, align 8
  store { ptr, ptr, i64, [3 x i64], [3 x i64] } %5, ptr %0, align 8
  %6 = load volatile { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %1, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %6, ptr %1, align 8
  %7 = load volatile { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %2, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, ptr %2, align 8
  %8 = load volatile { ptr, ptr, i64 }, ptr %3, align 8
  store { ptr, ptr, i64 } %8, ptr %3, align 8
  %9 = alloca i64, i64 1, align 8
  store i64 0, ptr %9, align 4, !tbaa !6
  call void @__catalyst__rt__device_init(ptr @"/home/pichau/QAgents/.venv/lib/python3.12/site-packages/pennylane_lightning/liblightning_gpu_catalyst.so", ptr @LightningGPUSimulator, ptr @"{}", i64 0, i1 false)
  %10 = call ptr @__catalyst__rt__qubit_allocate_array(i64 8)
  %11 = call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %10, i64 7)
  %12 = load ptr, ptr %11, align 8
  %13 = load i64, ptr %9, align 4, !tbaa !6
  %14 = add i64 %13, 1
  store i64 %14, ptr %9, align 4, !tbaa !6
  %15 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %16 = getelementptr inbounds double, ptr %15, i64 %13
  %17 = load double, ptr %16, align 8, !tbaa !1
  call void @__catalyst__qis__RY(double %17, ptr %12, ptr null)
  %18 = load i64, ptr %9, align 4, !tbaa !6
  %19 = add i64 %18, 1
  store i64 %19, ptr %9, align 4, !tbaa !6
  %20 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %21 = getelementptr inbounds double, ptr %20, i64 %18
  %22 = load double, ptr %21, align 8, !tbaa !1
  call void @__catalyst__qis__RZ(double %22, ptr %12, ptr null)
  %23 = load i64, ptr %9, align 4, !tbaa !6
  %24 = add i64 %23, 1
  store i64 %24, ptr %9, align 4, !tbaa !6
  %25 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %26 = getelementptr inbounds double, ptr %25, i64 %23
  %27 = load double, ptr %26, align 8, !tbaa !1
  call void @__catalyst__qis__RY(double %27, ptr %12, ptr null)
  %28 = load i64, ptr %9, align 4, !tbaa !6
  %29 = add i64 %28, 1
  store i64 %29, ptr %9, align 4, !tbaa !6
  %30 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %31 = getelementptr inbounds double, ptr %30, i64 %28
  %32 = load double, ptr %31, align 8, !tbaa !1
  call void @__catalyst__qis__RZ(double %32, ptr %12, ptr null)
  %33 = call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %10, i64 6)
  %34 = load ptr, ptr %33, align 8
  %35 = load i64, ptr %9, align 4, !tbaa !6
  %36 = add i64 %35, 1
  store i64 %36, ptr %9, align 4, !tbaa !6
  %37 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %38 = getelementptr inbounds double, ptr %37, i64 %35
  %39 = load double, ptr %38, align 8, !tbaa !1
  call void @__catalyst__qis__RY(double %39, ptr %34, ptr null)
  %40 = load i64, ptr %9, align 4, !tbaa !6
  %41 = add i64 %40, 1
  store i64 %41, ptr %9, align 4, !tbaa !6
  %42 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %43 = getelementptr inbounds double, ptr %42, i64 %40
  %44 = load double, ptr %43, align 8, !tbaa !1
  call void @__catalyst__qis__RZ(double %44, ptr %34, ptr null)
  %45 = load i64, ptr %9, align 4, !tbaa !6
  %46 = add i64 %45, 1
  store i64 %46, ptr %9, align 4, !tbaa !6
  %47 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %48 = getelementptr inbounds double, ptr %47, i64 %45
  %49 = load double, ptr %48, align 8, !tbaa !1
  call void @__catalyst__qis__RY(double %49, ptr %34, ptr null)
  %50 = load i64, ptr %9, align 4, !tbaa !6
  %51 = add i64 %50, 1
  store i64 %51, ptr %9, align 4, !tbaa !6
  %52 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %53 = getelementptr inbounds double, ptr %52, i64 %50
  %54 = load double, ptr %53, align 8, !tbaa !1
  call void @__catalyst__qis__RZ(double %54, ptr %34, ptr null)
  %55 = call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %10, i64 5)
  %56 = load ptr, ptr %55, align 8
  %57 = load i64, ptr %9, align 4, !tbaa !6
  %58 = add i64 %57, 1
  store i64 %58, ptr %9, align 4, !tbaa !6
  %59 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %60 = getelementptr inbounds double, ptr %59, i64 %57
  %61 = load double, ptr %60, align 8, !tbaa !1
  call void @__catalyst__qis__RY(double %61, ptr %56, ptr null)
  %62 = load i64, ptr %9, align 4, !tbaa !6
  %63 = add i64 %62, 1
  store i64 %63, ptr %9, align 4, !tbaa !6
  %64 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %65 = getelementptr inbounds double, ptr %64, i64 %62
  %66 = load double, ptr %65, align 8, !tbaa !1
  call void @__catalyst__qis__RZ(double %66, ptr %56, ptr null)
  %67 = load i64, ptr %9, align 4, !tbaa !6
  %68 = add i64 %67, 1
  store i64 %68, ptr %9, align 4, !tbaa !6
  %69 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %70 = getelementptr inbounds double, ptr %69, i64 %67
  %71 = load double, ptr %70, align 8, !tbaa !1
  call void @__catalyst__qis__RY(double %71, ptr %56, ptr null)
  %72 = load i64, ptr %9, align 4, !tbaa !6
  %73 = add i64 %72, 1
  store i64 %73, ptr %9, align 4, !tbaa !6
  %74 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %75 = getelementptr inbounds double, ptr %74, i64 %72
  %76 = load double, ptr %75, align 8, !tbaa !1
  call void @__catalyst__qis__RZ(double %76, ptr %56, ptr null)
  %77 = call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %10, i64 4)
  %78 = load ptr, ptr %77, align 8
  %79 = load i64, ptr %9, align 4, !tbaa !6
  %80 = add i64 %79, 1
  store i64 %80, ptr %9, align 4, !tbaa !6
  %81 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %82 = getelementptr inbounds double, ptr %81, i64 %79
  %83 = load double, ptr %82, align 8, !tbaa !1
  call void @__catalyst__qis__RY(double %83, ptr %78, ptr null)
  %84 = load i64, ptr %9, align 4, !tbaa !6
  %85 = add i64 %84, 1
  store i64 %85, ptr %9, align 4, !tbaa !6
  %86 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %87 = getelementptr inbounds double, ptr %86, i64 %84
  %88 = load double, ptr %87, align 8, !tbaa !1
  call void @__catalyst__qis__RZ(double %88, ptr %78, ptr null)
  %89 = load i64, ptr %9, align 4, !tbaa !6
  %90 = add i64 %89, 1
  store i64 %90, ptr %9, align 4, !tbaa !6
  %91 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %92 = getelementptr inbounds double, ptr %91, i64 %89
  %93 = load double, ptr %92, align 8, !tbaa !1
  call void @__catalyst__qis__RY(double %93, ptr %78, ptr null)
  %94 = load i64, ptr %9, align 4, !tbaa !6
  %95 = add i64 %94, 1
  store i64 %95, ptr %9, align 4, !tbaa !6
  %96 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %97 = getelementptr inbounds double, ptr %96, i64 %94
  %98 = load double, ptr %97, align 8, !tbaa !1
  call void @__catalyst__qis__RZ(double %98, ptr %78, ptr null)
  %99 = call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %10, i64 3)
  %100 = load ptr, ptr %99, align 8
  %101 = load i64, ptr %9, align 4, !tbaa !6
  %102 = add i64 %101, 1
  store i64 %102, ptr %9, align 4, !tbaa !6
  %103 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %104 = getelementptr inbounds double, ptr %103, i64 %101
  %105 = load double, ptr %104, align 8, !tbaa !1
  call void @__catalyst__qis__RY(double %105, ptr %100, ptr null)
  %106 = load i64, ptr %9, align 4, !tbaa !6
  %107 = add i64 %106, 1
  store i64 %107, ptr %9, align 4, !tbaa !6
  %108 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %109 = getelementptr inbounds double, ptr %108, i64 %106
  %110 = load double, ptr %109, align 8, !tbaa !1
  call void @__catalyst__qis__RZ(double %110, ptr %100, ptr null)
  %111 = load i64, ptr %9, align 4, !tbaa !6
  %112 = add i64 %111, 1
  store i64 %112, ptr %9, align 4, !tbaa !6
  %113 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %114 = getelementptr inbounds double, ptr %113, i64 %111
  %115 = load double, ptr %114, align 8, !tbaa !1
  call void @__catalyst__qis__RY(double %115, ptr %100, ptr null)
  %116 = load i64, ptr %9, align 4, !tbaa !6
  %117 = add i64 %116, 1
  store i64 %117, ptr %9, align 4, !tbaa !6
  %118 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %119 = getelementptr inbounds double, ptr %118, i64 %116
  %120 = load double, ptr %119, align 8, !tbaa !1
  call void @__catalyst__qis__RZ(double %120, ptr %100, ptr null)
  %121 = call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %10, i64 2)
  %122 = load ptr, ptr %121, align 8
  %123 = load i64, ptr %9, align 4, !tbaa !6
  %124 = add i64 %123, 1
  store i64 %124, ptr %9, align 4, !tbaa !6
  %125 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %126 = getelementptr inbounds double, ptr %125, i64 %123
  %127 = load double, ptr %126, align 8, !tbaa !1
  call void @__catalyst__qis__RY(double %127, ptr %122, ptr null)
  %128 = load i64, ptr %9, align 4, !tbaa !6
  %129 = add i64 %128, 1
  store i64 %129, ptr %9, align 4, !tbaa !6
  %130 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %131 = getelementptr inbounds double, ptr %130, i64 %128
  %132 = load double, ptr %131, align 8, !tbaa !1
  call void @__catalyst__qis__RZ(double %132, ptr %122, ptr null)
  %133 = load i64, ptr %9, align 4, !tbaa !6
  %134 = add i64 %133, 1
  store i64 %134, ptr %9, align 4, !tbaa !6
  %135 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %136 = getelementptr inbounds double, ptr %135, i64 %133
  %137 = load double, ptr %136, align 8, !tbaa !1
  call void @__catalyst__qis__RY(double %137, ptr %122, ptr null)
  %138 = load i64, ptr %9, align 4, !tbaa !6
  %139 = add i64 %138, 1
  store i64 %139, ptr %9, align 4, !tbaa !6
  %140 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %141 = getelementptr inbounds double, ptr %140, i64 %138
  %142 = load double, ptr %141, align 8, !tbaa !1
  call void @__catalyst__qis__RZ(double %142, ptr %122, ptr null)
  %143 = call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %10, i64 0)
  %144 = load ptr, ptr %143, align 8
  %145 = load i64, ptr %9, align 4, !tbaa !6
  %146 = add i64 %145, 1
  store i64 %146, ptr %9, align 4, !tbaa !6
  %147 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %148 = getelementptr inbounds double, ptr %147, i64 %145
  %149 = load double, ptr %148, align 8, !tbaa !1
  call void @__catalyst__qis__RY(double %149, ptr %144, ptr null)
  %150 = load i64, ptr %9, align 4, !tbaa !6
  %151 = add i64 %150, 1
  store i64 %151, ptr %9, align 4, !tbaa !6
  %152 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %153 = getelementptr inbounds double, ptr %152, i64 %150
  %154 = load double, ptr %153, align 8, !tbaa !1
  call void @__catalyst__qis__RZ(double %154, ptr %144, ptr null)
  %155 = load i64, ptr %9, align 4, !tbaa !6
  %156 = add i64 %155, 1
  store i64 %156, ptr %9, align 4, !tbaa !6
  %157 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %158 = getelementptr inbounds double, ptr %157, i64 %155
  %159 = load double, ptr %158, align 8, !tbaa !1
  call void @__catalyst__qis__RY(double %159, ptr %144, ptr null)
  %160 = load i64, ptr %9, align 4, !tbaa !6
  %161 = add i64 %160, 1
  store i64 %161, ptr %9, align 4, !tbaa !6
  %162 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %163 = getelementptr inbounds double, ptr %162, i64 %160
  %164 = load double, ptr %163, align 8, !tbaa !1
  call void @__catalyst__qis__RZ(double %164, ptr %144, ptr null)
  %165 = call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %10, i64 1)
  %166 = load ptr, ptr %165, align 8
  %167 = load i64, ptr %9, align 4, !tbaa !6
  %168 = add i64 %167, 1
  store i64 %168, ptr %9, align 4, !tbaa !6
  %169 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %170 = getelementptr inbounds double, ptr %169, i64 %167
  %171 = load double, ptr %170, align 8, !tbaa !1
  call void @__catalyst__qis__RY(double %171, ptr %166, ptr null)
  %172 = load i64, ptr %9, align 4, !tbaa !6
  %173 = add i64 %172, 1
  store i64 %173, ptr %9, align 4, !tbaa !6
  %174 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %175 = getelementptr inbounds double, ptr %174, i64 %172
  %176 = load double, ptr %175, align 8, !tbaa !1
  call void @__catalyst__qis__RZ(double %176, ptr %166, ptr null)
  %177 = load i64, ptr %9, align 4, !tbaa !6
  %178 = add i64 %177, 1
  store i64 %178, ptr %9, align 4, !tbaa !6
  %179 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %180 = getelementptr inbounds double, ptr %179, i64 %177
  %181 = load double, ptr %180, align 8, !tbaa !1
  call void @__catalyst__qis__RY(double %181, ptr %166, ptr null)
  %182 = load i64, ptr %9, align 4, !tbaa !6
  %183 = add i64 %182, 1
  store i64 %183, ptr %9, align 4, !tbaa !6
  %184 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %185 = getelementptr inbounds double, ptr %184, i64 %182
  %186 = load double, ptr %185, align 8, !tbaa !1
  call void @__catalyst__qis__RZ(double %186, ptr %166, ptr null)
  call void @__catalyst__qis__CNOT(ptr %144, ptr %166, ptr null)
  call void @__catalyst__qis__CNOT(ptr %166, ptr %122, ptr null)
  call void @__catalyst__qis__CNOT(ptr %122, ptr %100, ptr null)
  call void @__catalyst__qis__CNOT(ptr %100, ptr %78, ptr null)
  call void @__catalyst__qis__CNOT(ptr %78, ptr %56, ptr null)
  call void @__catalyst__qis__CNOT(ptr %56, ptr %34, ptr null)
  call void @__catalyst__qis__CNOT(ptr %34, ptr %12, ptr null)
  %187 = load i64, ptr %9, align 4, !tbaa !6
  %188 = add i64 %187, 1
  store i64 %188, ptr %9, align 4, !tbaa !6
  %189 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %190 = getelementptr inbounds double, ptr %189, i64 %187
  %191 = load double, ptr %190, align 8, !tbaa !1
  call void @__catalyst__qis__RZ(double %191, ptr %34, ptr null)
  %192 = load i64, ptr %9, align 4, !tbaa !6
  %193 = add i64 %192, 1
  store i64 %193, ptr %9, align 4, !tbaa !6
  %194 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %195 = getelementptr inbounds double, ptr %194, i64 %192
  %196 = load double, ptr %195, align 8, !tbaa !1
  call void @__catalyst__qis__RY(double %196, ptr %34, ptr null)
  %197 = load i64, ptr %9, align 4, !tbaa !6
  %198 = add i64 %197, 1
  store i64 %198, ptr %9, align 4, !tbaa !6
  %199 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %200 = getelementptr inbounds double, ptr %199, i64 %197
  %201 = load double, ptr %200, align 8, !tbaa !1
  call void @__catalyst__qis__RZ(double %201, ptr %34, ptr null)
  %202 = load i64, ptr %9, align 4, !tbaa !6
  %203 = add i64 %202, 1
  store i64 %203, ptr %9, align 4, !tbaa !6
  %204 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %205 = getelementptr inbounds double, ptr %204, i64 %202
  %206 = load double, ptr %205, align 8, !tbaa !1
  call void @__catalyst__qis__RZ(double %206, ptr %78, ptr null)
  %207 = load i64, ptr %9, align 4, !tbaa !6
  %208 = add i64 %207, 1
  store i64 %208, ptr %9, align 4, !tbaa !6
  %209 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %210 = getelementptr inbounds double, ptr %209, i64 %207
  %211 = load double, ptr %210, align 8, !tbaa !1
  call void @__catalyst__qis__RY(double %211, ptr %78, ptr null)
  %212 = load i64, ptr %9, align 4, !tbaa !6
  %213 = add i64 %212, 1
  store i64 %213, ptr %9, align 4, !tbaa !6
  %214 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %215 = getelementptr inbounds double, ptr %214, i64 %212
  %216 = load double, ptr %215, align 8, !tbaa !1
  call void @__catalyst__qis__RZ(double %216, ptr %78, ptr null)
  call void @__catalyst__qis__CNOT(ptr %12, ptr %144, ptr null)
  %217 = load i64, ptr %9, align 4, !tbaa !6
  %218 = add i64 %217, 1
  store i64 %218, ptr %9, align 4, !tbaa !6
  %219 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %220 = getelementptr inbounds double, ptr %219, i64 %217
  %221 = load double, ptr %220, align 8, !tbaa !1
  call void @__catalyst__qis__RZ(double %221, ptr %144, ptr null)
  %222 = load i64, ptr %9, align 4, !tbaa !6
  %223 = add i64 %222, 1
  store i64 %223, ptr %9, align 4, !tbaa !6
  %224 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %225 = getelementptr inbounds double, ptr %224, i64 %222
  %226 = load double, ptr %225, align 8, !tbaa !1
  call void @__catalyst__qis__RY(double %226, ptr %144, ptr null)
  %227 = load i64, ptr %9, align 4, !tbaa !6
  %228 = add i64 %227, 1
  store i64 %228, ptr %9, align 4, !tbaa !6
  %229 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %230 = getelementptr inbounds double, ptr %229, i64 %227
  %231 = load double, ptr %230, align 8, !tbaa !1
  call void @__catalyst__qis__RZ(double %231, ptr %144, ptr null)
  %232 = load i64, ptr %9, align 4, !tbaa !6
  %233 = add i64 %232, 1
  store i64 %233, ptr %9, align 4, !tbaa !6
  %234 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %235 = getelementptr inbounds double, ptr %234, i64 %232
  %236 = load double, ptr %235, align 8, !tbaa !1
  call void @__catalyst__qis__RZ(double %236, ptr %122, ptr null)
  %237 = load i64, ptr %9, align 4, !tbaa !6
  %238 = add i64 %237, 1
  store i64 %238, ptr %9, align 4, !tbaa !6
  %239 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %240 = getelementptr inbounds double, ptr %239, i64 %237
  %241 = load double, ptr %240, align 8, !tbaa !1
  call void @__catalyst__qis__RY(double %241, ptr %122, ptr null)
  %242 = load i64, ptr %9, align 4, !tbaa !6
  %243 = add i64 %242, 1
  store i64 %243, ptr %9, align 4, !tbaa !6
  %244 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %245 = getelementptr inbounds double, ptr %244, i64 %242
  %246 = load double, ptr %245, align 8, !tbaa !1
  call void @__catalyst__qis__RZ(double %246, ptr %122, ptr null)
  call void @__catalyst__qis__CNOT(ptr %144, ptr %122, ptr null)
  call void @__catalyst__qis__CNOT(ptr %122, ptr %78, ptr null)
  call void @__catalyst__qis__CNOT(ptr %78, ptr %34, ptr null)
  call void @__catalyst__qis__CNOT(ptr %34, ptr %144, ptr null)
  %247 = load i64, ptr %9, align 4, !tbaa !6
  %248 = add i64 %247, 1
  store i64 %248, ptr %9, align 4, !tbaa !6
  %249 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %250 = getelementptr inbounds double, ptr %249, i64 %247
  %251 = load double, ptr %250, align 8, !tbaa !1
  call void @__catalyst__qis__RZ(double %251, ptr %144, ptr null)
  %252 = load i64, ptr %9, align 4, !tbaa !6
  %253 = add i64 %252, 1
  store i64 %253, ptr %9, align 4, !tbaa !6
  %254 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %255 = getelementptr inbounds double, ptr %254, i64 %252
  %256 = load double, ptr %255, align 8, !tbaa !1
  call void @__catalyst__qis__RY(double %256, ptr %144, ptr null)
  %257 = load i64, ptr %9, align 4, !tbaa !6
  %258 = add i64 %257, 1
  store i64 %258, ptr %9, align 4, !tbaa !6
  %259 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %260 = getelementptr inbounds double, ptr %259, i64 %257
  %261 = load double, ptr %260, align 8, !tbaa !1
  call void @__catalyst__qis__RZ(double %261, ptr %144, ptr null)
  %262 = load i64, ptr %9, align 4, !tbaa !6
  %263 = add i64 %262, 1
  store i64 %263, ptr %9, align 4, !tbaa !6
  %264 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %265 = getelementptr inbounds double, ptr %264, i64 %262
  %266 = load double, ptr %265, align 8, !tbaa !1
  call void @__catalyst__qis__RZ(double %266, ptr %56, ptr null)
  %267 = load i64, ptr %9, align 4, !tbaa !6
  %268 = add i64 %267, 1
  store i64 %268, ptr %9, align 4, !tbaa !6
  %269 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %270 = getelementptr inbounds double, ptr %269, i64 %267
  %271 = load double, ptr %270, align 8, !tbaa !1
  call void @__catalyst__qis__RY(double %271, ptr %56, ptr null)
  %272 = load i64, ptr %9, align 4, !tbaa !6
  %273 = add i64 %272, 1
  store i64 %273, ptr %9, align 4, !tbaa !6
  %274 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %275 = getelementptr inbounds double, ptr %274, i64 %272
  %276 = load double, ptr %275, align 8, !tbaa !1
  call void @__catalyst__qis__RZ(double %276, ptr %56, ptr null)
  %277 = load i64, ptr %9, align 4, !tbaa !6
  %278 = add i64 %277, 1
  store i64 %278, ptr %9, align 4, !tbaa !6
  %279 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %280 = getelementptr inbounds double, ptr %279, i64 %277
  %281 = load double, ptr %280, align 8, !tbaa !1
  call void @__catalyst__qis__RZ(double %281, ptr %166, ptr null)
  %282 = load i64, ptr %9, align 4, !tbaa !6
  %283 = add i64 %282, 1
  store i64 %283, ptr %9, align 4, !tbaa !6
  %284 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %285 = getelementptr inbounds double, ptr %284, i64 %282
  %286 = load double, ptr %285, align 8, !tbaa !1
  call void @__catalyst__qis__RY(double %286, ptr %166, ptr null)
  %287 = load i64, ptr %9, align 4, !tbaa !6
  %288 = add i64 %287, 1
  store i64 %288, ptr %9, align 4, !tbaa !6
  %289 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %290 = getelementptr inbounds double, ptr %289, i64 %287
  %291 = load double, ptr %290, align 8, !tbaa !1
  call void @__catalyst__qis__RZ(double %291, ptr %166, ptr null)
  %292 = load i64, ptr %9, align 4, !tbaa !6
  %293 = add i64 %292, 1
  store i64 %293, ptr %9, align 4, !tbaa !6
  %294 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %295 = getelementptr inbounds double, ptr %294, i64 %292
  %296 = load double, ptr %295, align 8, !tbaa !1
  call void @__catalyst__qis__RZ(double %296, ptr %100, ptr null)
  %297 = load i64, ptr %9, align 4, !tbaa !6
  %298 = add i64 %297, 1
  store i64 %298, ptr %9, align 4, !tbaa !6
  %299 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %300 = getelementptr inbounds double, ptr %299, i64 %297
  %301 = load double, ptr %300, align 8, !tbaa !1
  call void @__catalyst__qis__RY(double %301, ptr %100, ptr null)
  %302 = load i64, ptr %9, align 4, !tbaa !6
  %303 = add i64 %302, 1
  store i64 %303, ptr %9, align 4, !tbaa !6
  %304 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %305 = getelementptr inbounds double, ptr %304, i64 %302
  %306 = load double, ptr %305, align 8, !tbaa !1
  call void @__catalyst__qis__RZ(double %306, ptr %100, ptr null)
  call void @__catalyst__qis__CNOT(ptr %166, ptr %100, ptr null)
  call void @__catalyst__qis__CNOT(ptr %100, ptr %56, ptr null)
  %307 = load i64, ptr %9, align 4, !tbaa !6
  %308 = add i64 %307, 1
  store i64 %308, ptr %9, align 4, !tbaa !6
  %309 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %310 = getelementptr inbounds double, ptr %309, i64 %307
  %311 = load double, ptr %310, align 8, !tbaa !1
  call void @__catalyst__qis__RZ(double %311, ptr %100, ptr null)
  %312 = load i64, ptr %9, align 4, !tbaa !6
  %313 = add i64 %312, 1
  store i64 %313, ptr %9, align 4, !tbaa !6
  %314 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %315 = getelementptr inbounds double, ptr %314, i64 %312
  %316 = load double, ptr %315, align 8, !tbaa !1
  call void @__catalyst__qis__RY(double %316, ptr %100, ptr null)
  %317 = load i64, ptr %9, align 4, !tbaa !6
  %318 = add i64 %317, 1
  store i64 %318, ptr %9, align 4, !tbaa !6
  %319 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %320 = getelementptr inbounds double, ptr %319, i64 %317
  %321 = load double, ptr %320, align 8, !tbaa !1
  call void @__catalyst__qis__RZ(double %321, ptr %100, ptr null)
  call void @__catalyst__qis__CNOT(ptr %144, ptr %100, ptr null)
  %322 = load i64, ptr %9, align 4, !tbaa !6
  %323 = add i64 %322, 1
  store i64 %323, ptr %9, align 4, !tbaa !6
  %324 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %325 = getelementptr inbounds double, ptr %324, i64 %322
  %326 = load double, ptr %325, align 8, !tbaa !1
  call void @__catalyst__qis__RZ(double %326, ptr %122, ptr null)
  %327 = load i64, ptr %9, align 4, !tbaa !6
  %328 = add i64 %327, 1
  store i64 %328, ptr %9, align 4, !tbaa !6
  %329 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %330 = getelementptr inbounds double, ptr %329, i64 %327
  %331 = load double, ptr %330, align 8, !tbaa !1
  call void @__catalyst__qis__RY(double %331, ptr %122, ptr null)
  %332 = load i64, ptr %9, align 4, !tbaa !6
  %333 = add i64 %332, 1
  store i64 %333, ptr %9, align 4, !tbaa !6
  %334 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %335 = getelementptr inbounds double, ptr %334, i64 %332
  %336 = load double, ptr %335, align 8, !tbaa !1
  call void @__catalyst__qis__RZ(double %336, ptr %122, ptr null)
  %337 = load i64, ptr %9, align 4, !tbaa !6
  %338 = add i64 %337, 1
  store i64 %338, ptr %9, align 4, !tbaa !6
  %339 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %340 = getelementptr inbounds double, ptr %339, i64 %337
  %341 = load double, ptr %340, align 8, !tbaa !1
  call void @__catalyst__qis__RZ(double %341, ptr %12, ptr null)
  %342 = load i64, ptr %9, align 4, !tbaa !6
  %343 = add i64 %342, 1
  store i64 %343, ptr %9, align 4, !tbaa !6
  %344 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %345 = getelementptr inbounds double, ptr %344, i64 %342
  %346 = load double, ptr %345, align 8, !tbaa !1
  call void @__catalyst__qis__RY(double %346, ptr %12, ptr null)
  %347 = load i64, ptr %9, align 4, !tbaa !6
  %348 = add i64 %347, 1
  store i64 %348, ptr %9, align 4, !tbaa !6
  %349 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %350 = getelementptr inbounds double, ptr %349, i64 %347
  %351 = load double, ptr %350, align 8, !tbaa !1
  call void @__catalyst__qis__RZ(double %351, ptr %12, ptr null)
  call void @__catalyst__qis__CNOT(ptr %56, ptr %12, ptr null)
  %352 = load i64, ptr %9, align 4, !tbaa !6
  %353 = add i64 %352, 1
  store i64 %353, ptr %9, align 4, !tbaa !6
  %354 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %355 = getelementptr inbounds double, ptr %354, i64 %352
  %356 = load double, ptr %355, align 8, !tbaa !1
  call void @__catalyst__qis__RZ(double %356, ptr %56, ptr null)
  %357 = load i64, ptr %9, align 4, !tbaa !6
  %358 = add i64 %357, 1
  store i64 %358, ptr %9, align 4, !tbaa !6
  %359 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %360 = getelementptr inbounds double, ptr %359, i64 %357
  %361 = load double, ptr %360, align 8, !tbaa !1
  call void @__catalyst__qis__RY(double %361, ptr %56, ptr null)
  %362 = load i64, ptr %9, align 4, !tbaa !6
  %363 = add i64 %362, 1
  store i64 %363, ptr %9, align 4, !tbaa !6
  %364 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %365 = getelementptr inbounds double, ptr %364, i64 %362
  %366 = load double, ptr %365, align 8, !tbaa !1
  call void @__catalyst__qis__RZ(double %366, ptr %56, ptr null)
  call void @__catalyst__qis__CNOT(ptr %122, ptr %56, ptr null)
  call void @__catalyst__qis__CNOT(ptr %56, ptr %144, ptr null)
  %367 = load i64, ptr %9, align 4, !tbaa !6
  %368 = add i64 %367, 1
  store i64 %368, ptr %9, align 4, !tbaa !6
  %369 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %370 = getelementptr inbounds double, ptr %369, i64 %367
  %371 = load double, ptr %370, align 8, !tbaa !1
  call void @__catalyst__qis__RZ(double %371, ptr %144, ptr null)
  %372 = load i64, ptr %9, align 4, !tbaa !6
  %373 = add i64 %372, 1
  store i64 %373, ptr %9, align 4, !tbaa !6
  %374 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %375 = getelementptr inbounds double, ptr %374, i64 %372
  %376 = load double, ptr %375, align 8, !tbaa !1
  call void @__catalyst__qis__RY(double %376, ptr %144, ptr null)
  %377 = load i64, ptr %9, align 4, !tbaa !6
  %378 = add i64 %377, 1
  store i64 %378, ptr %9, align 4, !tbaa !6
  %379 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %380 = getelementptr inbounds double, ptr %379, i64 %377
  %381 = load double, ptr %380, align 8, !tbaa !1
  call void @__catalyst__qis__RZ(double %381, ptr %144, ptr null)
  call void @__catalyst__qis__CNOT(ptr %12, ptr %166, ptr null)
  %382 = load i64, ptr %9, align 4, !tbaa !6
  %383 = add i64 %382, 1
  store i64 %383, ptr %9, align 4, !tbaa !6
  %384 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %385 = getelementptr inbounds double, ptr %384, i64 %382
  %386 = load double, ptr %385, align 8, !tbaa !1
  call void @__catalyst__qis__RZ(double %386, ptr %12, ptr null)
  %387 = load i64, ptr %9, align 4, !tbaa !6
  %388 = add i64 %387, 1
  store i64 %388, ptr %9, align 4, !tbaa !6
  %389 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %390 = getelementptr inbounds double, ptr %389, i64 %387
  %391 = load double, ptr %390, align 8, !tbaa !1
  call void @__catalyst__qis__RY(double %391, ptr %12, ptr null)
  %392 = load i64, ptr %9, align 4, !tbaa !6
  %393 = add i64 %392, 1
  store i64 %393, ptr %9, align 4, !tbaa !6
  %394 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %395 = getelementptr inbounds double, ptr %394, i64 %392
  %396 = load double, ptr %395, align 8, !tbaa !1
  call void @__catalyst__qis__RZ(double %396, ptr %12, ptr null)
  %397 = load i64, ptr %9, align 4, !tbaa !6
  %398 = add i64 %397, 1
  store i64 %398, ptr %9, align 4, !tbaa !6
  %399 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %400 = getelementptr inbounds double, ptr %399, i64 %397
  %401 = load double, ptr %400, align 8, !tbaa !1
  call void @__catalyst__qis__RZ(double %401, ptr %166, ptr null)
  %402 = load i64, ptr %9, align 4, !tbaa !6
  %403 = add i64 %402, 1
  store i64 %403, ptr %9, align 4, !tbaa !6
  %404 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %405 = getelementptr inbounds double, ptr %404, i64 %402
  %406 = load double, ptr %405, align 8, !tbaa !1
  call void @__catalyst__qis__RY(double %406, ptr %166, ptr null)
  %407 = load i64, ptr %9, align 4, !tbaa !6
  %408 = add i64 %407, 1
  store i64 %408, ptr %9, align 4, !tbaa !6
  %409 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %410 = getelementptr inbounds double, ptr %409, i64 %407
  %411 = load double, ptr %410, align 8, !tbaa !1
  call void @__catalyst__qis__RZ(double %411, ptr %166, ptr null)
  %412 = load i64, ptr %9, align 4, !tbaa !6
  %413 = add i64 %412, 1
  store i64 %413, ptr %9, align 4, !tbaa !6
  %414 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %415 = getelementptr inbounds double, ptr %414, i64 %412
  %416 = load double, ptr %415, align 8, !tbaa !1
  call void @__catalyst__qis__RZ(double %416, ptr %78, ptr null)
  %417 = load i64, ptr %9, align 4, !tbaa !6
  %418 = add i64 %417, 1
  store i64 %418, ptr %9, align 4, !tbaa !6
  %419 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %420 = getelementptr inbounds double, ptr %419, i64 %417
  %421 = load double, ptr %420, align 8, !tbaa !1
  call void @__catalyst__qis__RY(double %421, ptr %78, ptr null)
  %422 = load i64, ptr %9, align 4, !tbaa !6
  %423 = add i64 %422, 1
  store i64 %423, ptr %9, align 4, !tbaa !6
  %424 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %425 = getelementptr inbounds double, ptr %424, i64 %422
  %426 = load double, ptr %425, align 8, !tbaa !1
  call void @__catalyst__qis__RZ(double %426, ptr %78, ptr null)
  call void @__catalyst__qis__CNOT(ptr %166, ptr %78, ptr null)
  call void @__catalyst__qis__CNOT(ptr %78, ptr %12, ptr null)
  %427 = load i64, ptr %9, align 4, !tbaa !6
  %428 = add i64 %427, 1
  store i64 %428, ptr %9, align 4, !tbaa !6
  %429 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %430 = getelementptr inbounds double, ptr %429, i64 %427
  %431 = load double, ptr %430, align 8, !tbaa !1
  call void @__catalyst__qis__RZ(double %431, ptr %78, ptr null)
  %432 = load i64, ptr %9, align 4, !tbaa !6
  %433 = add i64 %432, 1
  store i64 %433, ptr %9, align 4, !tbaa !6
  %434 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %435 = getelementptr inbounds double, ptr %434, i64 %432
  %436 = load double, ptr %435, align 8, !tbaa !1
  call void @__catalyst__qis__RY(double %436, ptr %78, ptr null)
  %437 = load i64, ptr %9, align 4, !tbaa !6
  %438 = add i64 %437, 1
  store i64 %438, ptr %9, align 4, !tbaa !6
  %439 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %440 = getelementptr inbounds double, ptr %439, i64 %437
  %441 = load double, ptr %440, align 8, !tbaa !1
  call void @__catalyst__qis__RZ(double %441, ptr %78, ptr null)
  call void @__catalyst__qis__CNOT(ptr %144, ptr %78, ptr null)
  call void @__catalyst__qis__CNOT(ptr %78, ptr %144, ptr null)
  %442 = load i64, ptr %9, align 4, !tbaa !6
  %443 = add i64 %442, 1
  store i64 %443, ptr %9, align 4, !tbaa !6
  %444 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %445 = getelementptr inbounds double, ptr %444, i64 %442
  %446 = load double, ptr %445, align 8, !tbaa !1
  call void @__catalyst__qis__RZ(double %446, ptr %34, ptr null)
  %447 = load i64, ptr %9, align 4, !tbaa !6
  %448 = add i64 %447, 1
  store i64 %448, ptr %9, align 4, !tbaa !6
  %449 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %450 = getelementptr inbounds double, ptr %449, i64 %447
  %451 = load double, ptr %450, align 8, !tbaa !1
  call void @__catalyst__qis__RY(double %451, ptr %34, ptr null)
  %452 = load i64, ptr %9, align 4, !tbaa !6
  %453 = add i64 %452, 1
  store i64 %453, ptr %9, align 4, !tbaa !6
  %454 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %455 = getelementptr inbounds double, ptr %454, i64 %452
  %456 = load double, ptr %455, align 8, !tbaa !1
  call void @__catalyst__qis__RZ(double %456, ptr %34, ptr null)
  call void @__catalyst__qis__CNOT(ptr %100, ptr %34, ptr null)
  call void @__catalyst__qis__CNOT(ptr %34, ptr %166, ptr null)
  %457 = load i64, ptr %9, align 4, !tbaa !6
  %458 = add i64 %457, 1
  store i64 %458, ptr %9, align 4, !tbaa !6
  %459 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %460 = getelementptr inbounds double, ptr %459, i64 %457
  %461 = load double, ptr %460, align 8, !tbaa !1
  call void @__catalyst__qis__RZ(double %461, ptr %166, ptr null)
  %462 = load i64, ptr %9, align 4, !tbaa !6
  %463 = add i64 %462, 1
  store i64 %463, ptr %9, align 4, !tbaa !6
  %464 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %465 = getelementptr inbounds double, ptr %464, i64 %462
  %466 = load double, ptr %465, align 8, !tbaa !1
  call void @__catalyst__qis__RY(double %466, ptr %166, ptr null)
  %467 = load i64, ptr %9, align 4, !tbaa !6
  %468 = add i64 %467, 1
  store i64 %468, ptr %9, align 4, !tbaa !6
  %469 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %470 = getelementptr inbounds double, ptr %469, i64 %467
  %471 = load double, ptr %470, align 8, !tbaa !1
  call void @__catalyst__qis__RZ(double %471, ptr %166, ptr null)
  %472 = load i64, ptr %9, align 4, !tbaa !6
  %473 = add i64 %472, 1
  store i64 %473, ptr %9, align 4, !tbaa !6
  %474 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %475 = getelementptr inbounds double, ptr %474, i64 %472
  %476 = load double, ptr %475, align 8, !tbaa !1
  call void @__catalyst__qis__RZ(double %476, ptr %56, ptr null)
  %477 = load i64, ptr %9, align 4, !tbaa !6
  %478 = add i64 %477, 1
  store i64 %478, ptr %9, align 4, !tbaa !6
  %479 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %480 = getelementptr inbounds double, ptr %479, i64 %477
  %481 = load double, ptr %480, align 8, !tbaa !1
  call void @__catalyst__qis__RY(double %481, ptr %56, ptr null)
  %482 = load i64, ptr %9, align 4, !tbaa !6
  %483 = add i64 %482, 1
  store i64 %483, ptr %9, align 4, !tbaa !6
  %484 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %485 = getelementptr inbounds double, ptr %484, i64 %482
  %486 = load double, ptr %485, align 8, !tbaa !1
  call void @__catalyst__qis__RZ(double %486, ptr %56, ptr null)
  call void @__catalyst__qis__CNOT(ptr %166, ptr %56, ptr null)
  call void @__catalyst__qis__CNOT(ptr %56, ptr %166, ptr null)
  call void @__catalyst__qis__CNOT(ptr %12, ptr %122, ptr null)
  %487 = load i64, ptr %9, align 4, !tbaa !6
  %488 = add i64 %487, 1
  store i64 %488, ptr %9, align 4, !tbaa !6
  %489 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %490 = getelementptr inbounds double, ptr %489, i64 %487
  %491 = load double, ptr %490, align 8, !tbaa !1
  call void @__catalyst__qis__RZ(double %491, ptr %122, ptr null)
  %492 = load i64, ptr %9, align 4, !tbaa !6
  %493 = add i64 %492, 1
  store i64 %493, ptr %9, align 4, !tbaa !6
  %494 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %495 = getelementptr inbounds double, ptr %494, i64 %492
  %496 = load double, ptr %495, align 8, !tbaa !1
  call void @__catalyst__qis__RY(double %496, ptr %122, ptr null)
  %497 = load i64, ptr %9, align 4, !tbaa !6
  %498 = add i64 %497, 1
  store i64 %498, ptr %9, align 4, !tbaa !6
  %499 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %500 = getelementptr inbounds double, ptr %499, i64 %497
  %501 = load double, ptr %500, align 8, !tbaa !1
  call void @__catalyst__qis__RZ(double %501, ptr %122, ptr null)
  %502 = load i64, ptr %9, align 4, !tbaa !6
  %503 = add i64 %502, 1
  store i64 %503, ptr %9, align 4, !tbaa !6
  %504 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %505 = getelementptr inbounds double, ptr %504, i64 %502
  %506 = load double, ptr %505, align 8, !tbaa !1
  call void @__catalyst__qis__RZ(double %506, ptr %34, ptr null)
  %507 = load i64, ptr %9, align 4, !tbaa !6
  %508 = add i64 %507, 1
  store i64 %508, ptr %9, align 4, !tbaa !6
  %509 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %510 = getelementptr inbounds double, ptr %509, i64 %507
  %511 = load double, ptr %510, align 8, !tbaa !1
  call void @__catalyst__qis__RY(double %511, ptr %34, ptr null)
  %512 = load i64, ptr %9, align 4, !tbaa !6
  %513 = add i64 %512, 1
  store i64 %513, ptr %9, align 4, !tbaa !6
  %514 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %515 = getelementptr inbounds double, ptr %514, i64 %512
  %516 = load double, ptr %515, align 8, !tbaa !1
  call void @__catalyst__qis__RZ(double %516, ptr %34, ptr null)
  call void @__catalyst__qis__CNOT(ptr %122, ptr %34, ptr null)
  call void @__catalyst__qis__CNOT(ptr %34, ptr %122, ptr null)
  %517 = load i64, ptr %9, align 4, !tbaa !6
  %518 = add i64 %517, 1
  store i64 %518, ptr %9, align 4, !tbaa !6
  %519 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %520 = getelementptr inbounds double, ptr %519, i64 %517
  %521 = load double, ptr %520, align 8, !tbaa !1
  call void @__catalyst__qis__RZ(double %521, ptr %100, ptr null)
  %522 = load i64, ptr %9, align 4, !tbaa !6
  %523 = add i64 %522, 1
  store i64 %523, ptr %9, align 4, !tbaa !6
  %524 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %525 = getelementptr inbounds double, ptr %524, i64 %522
  %526 = load double, ptr %525, align 8, !tbaa !1
  call void @__catalyst__qis__RY(double %526, ptr %100, ptr null)
  %527 = load i64, ptr %9, align 4, !tbaa !6
  %528 = add i64 %527, 1
  store i64 %528, ptr %9, align 4, !tbaa !6
  %529 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %530 = getelementptr inbounds double, ptr %529, i64 %527
  %531 = load double, ptr %530, align 8, !tbaa !1
  call void @__catalyst__qis__RZ(double %531, ptr %100, ptr null)
  %532 = load i64, ptr %9, align 4, !tbaa !6
  %533 = add i64 %532, 1
  store i64 %533, ptr %9, align 4, !tbaa !6
  %534 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %535 = getelementptr inbounds double, ptr %534, i64 %532
  %536 = load double, ptr %535, align 8, !tbaa !1
  call void @__catalyst__qis__RZ(double %536, ptr %12, ptr null)
  %537 = load i64, ptr %9, align 4, !tbaa !6
  %538 = add i64 %537, 1
  store i64 %538, ptr %9, align 4, !tbaa !6
  %539 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %540 = getelementptr inbounds double, ptr %539, i64 %537
  %541 = load double, ptr %540, align 8, !tbaa !1
  call void @__catalyst__qis__RY(double %541, ptr %12, ptr null)
  %542 = load i64, ptr %9, align 4, !tbaa !6
  %543 = add i64 %542, 1
  store i64 %543, ptr %9, align 4, !tbaa !6
  %544 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %545 = getelementptr inbounds double, ptr %544, i64 %542
  %546 = load double, ptr %545, align 8, !tbaa !1
  call void @__catalyst__qis__RZ(double %546, ptr %12, ptr null)
  call void @__catalyst__qis__CNOT(ptr %100, ptr %12, ptr null)
  call void @__catalyst__qis__CNOT(ptr %12, ptr %100, ptr null)
  %547 = call i64 @__catalyst__qis__NamedObs(i64 3, ptr %144)
  %548 = call double @__catalyst__qis__Expval(i64 %547)
  %549 = call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %550 = ptrtoint ptr %549 to i64
  %551 = add i64 %550, 63
  %552 = urem i64 %551, 64
  %553 = sub i64 %551, %552
  %554 = inttoptr i64 %553 to ptr
  store double %548, ptr %554, align 8, !tbaa !1
  call void @__catalyst__rt__qubit_release_array(ptr %10)
  call void @__catalyst__rt__device_release()
  %555 = load double, ptr %554, align 8, !tbaa !1
  %556 = extractvalue { ptr, ptr, i64 } %8, 1
  store double %555, ptr %556, align 8, !tbaa !1
  ret void
}

define ptr @qnode_forward_0.quantum.augfwd(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7) {
  call void @qnode_forward_0.quantum(ptr %0, ptr %2, ptr %4, ptr %6)
  ret ptr null
}

define void @qnode_forward_0.preprocess(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, ptr %9, ptr %10, i64 %11, i64 %12, i64 %13, i64 %14, ptr %15, ptr %16, i64 %17) {
  %19 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %9, 0
  %20 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %19, ptr %10, 1
  %21 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, i64 %11, 2
  %22 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %21, i64 %12, 3, 0
  %23 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %22, i64 %13, 4, 0
  %24 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } poison, ptr %0, 0
  %25 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %24, ptr %1, 1
  %26 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %25, i64 %2, 2
  %27 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %26, i64 %3, 3, 0
  %28 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %27, i64 %6, 4, 0
  %29 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %28, i64 %4, 3, 1
  %30 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %29, i64 %7, 4, 1
  %31 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %30, i64 %5, 3, 2
  %32 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %31, i64 %8, 4, 2
  %33 = alloca { ptr, ptr, i64 }, i64 1, align 8
  %34 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, i64 1, align 8
  %35 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, i64 1, align 8
  %36 = alloca { ptr, ptr, i64, [3 x i64], [3 x i64] }, i64 1, align 8
  %37 = getelementptr double, ptr null, i64 %14
  %38 = ptrtoint ptr %37 to i64
  %39 = call ptr @_mlir_memref_to_llvm_alloc(i64 %38)
  %40 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %39, 0
  %41 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %40, ptr %39, 1
  %42 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %41, i64 0, 2
  %43 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %42, i64 %14, 3, 0
  %44 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %43, i64 1, 4, 0
  %45 = alloca i64, i64 1, align 8
  store i64 0, ptr %45, align 4, !tbaa !6
  %46 = getelementptr inbounds float, ptr %1, i32 74
  %47 = load float, ptr %46, align 4, !tbaa !4
  %48 = getelementptr inbounds float, ptr %1, i32 73
  %49 = load float, ptr %48, align 4, !tbaa !4
  %50 = getelementptr inbounds float, ptr %1, i32 72
  %51 = load float, ptr %50, align 4, !tbaa !4
  %52 = getelementptr inbounds float, ptr %1, i32 50
  %53 = load float, ptr %52, align 4, !tbaa !4
  %54 = getelementptr inbounds float, ptr %1, i32 49
  %55 = load float, ptr %54, align 4, !tbaa !4
  %56 = getelementptr inbounds float, ptr %1, i32 48
  %57 = load float, ptr %56, align 4, !tbaa !4
  %58 = getelementptr inbounds float, ptr %1, i32 44
  %59 = load float, ptr %58, align 4, !tbaa !4
  %60 = getelementptr inbounds float, ptr %1, i32 43
  %61 = load float, ptr %60, align 4, !tbaa !4
  %62 = getelementptr inbounds float, ptr %1, i32 42
  %63 = load float, ptr %62, align 4, !tbaa !4
  %64 = getelementptr inbounds float, ptr %1, i32 23
  %65 = load float, ptr %64, align 4, !tbaa !4
  %66 = getelementptr inbounds float, ptr %1, i32 22
  %67 = load float, ptr %66, align 4, !tbaa !4
  %68 = getelementptr inbounds float, ptr %1, i32 21
  %69 = load float, ptr %68, align 4, !tbaa !4
  %70 = call ptr @_mlir_memref_to_llvm_alloc(i64 96)
  %71 = ptrtoint ptr %70 to i64
  %72 = add i64 %71, 63
  %73 = urem i64 %72, 64
  %74 = sub i64 %72, %73
  %75 = inttoptr i64 %74 to ptr
  br label %76

76:                                               ; preds = %79, %18
  %77 = phi i64 [ %82, %79 ], [ 0, %18 ]
  %78 = icmp slt i64 %77, 8
  br i1 %78, label %79, label %83

79:                                               ; preds = %76
  %80 = load float, ptr @__constant_xf32, align 4, !tbaa !4
  %81 = getelementptr inbounds float, ptr %75, i64 %77
  store float %80, ptr %81, align 4, !tbaa !4
  %82 = add i64 %77, 1
  br label %76

83:                                               ; preds = %76
  br label %84

84:                                               ; preds = %87, %83
  %85 = phi i64 [ %94, %87 ], [ 0, %83 ]
  %86 = icmp slt i64 %85, 8
  br i1 %86, label %87, label %95

87:                                               ; preds = %84
  %88 = getelementptr inbounds float, ptr %75, i64 %85
  %89 = load float, ptr %88, align 4, !tbaa !4
  %90 = getelementptr inbounds float, ptr %10, i64 %85
  %91 = load float, ptr %90, align 4, !tbaa !4
  %92 = fmul float %89, %91
  %93 = getelementptr inbounds float, ptr %75, i64 %85
  store float %92, ptr %93, align 4, !tbaa !4
  %94 = add i64 %85, 1
  br label %84

95:                                               ; preds = %84
  %96 = getelementptr inbounds float, ptr %75, i32 7
  %97 = load float, ptr %96, align 4, !tbaa !4
  %98 = fpext float %97 to double
  %99 = load i64, ptr %45, align 4, !tbaa !6
  %100 = getelementptr inbounds double, ptr %39, i64 %99
  store double %98, ptr %100, align 8, !tbaa !1
  %101 = add i64 %99, 1
  store i64 %101, ptr %45, align 4, !tbaa !6
  %102 = fpext float %69 to double
  %103 = load i64, ptr %45, align 4, !tbaa !6
  %104 = getelementptr inbounds double, ptr %39, i64 %103
  store double %102, ptr %104, align 8, !tbaa !1
  %105 = add i64 %103, 1
  store i64 %105, ptr %45, align 4, !tbaa !6
  %106 = fpext float %67 to double
  %107 = load i64, ptr %45, align 4, !tbaa !6
  %108 = getelementptr inbounds double, ptr %39, i64 %107
  store double %106, ptr %108, align 8, !tbaa !1
  %109 = add i64 %107, 1
  store i64 %109, ptr %45, align 4, !tbaa !6
  %110 = fpext float %65 to double
  %111 = load i64, ptr %45, align 4, !tbaa !6
  %112 = getelementptr inbounds double, ptr %39, i64 %111
  store double %110, ptr %112, align 8, !tbaa !1
  %113 = add i64 %111, 1
  store i64 %113, ptr %45, align 4, !tbaa !6
  %114 = getelementptr inbounds float, ptr %1, i32 20
  %115 = load float, ptr %114, align 4, !tbaa !4
  %116 = getelementptr inbounds float, ptr %1, i32 19
  %117 = load float, ptr %116, align 4, !tbaa !4
  %118 = getelementptr inbounds float, ptr %1, i32 18
  %119 = load float, ptr %118, align 4, !tbaa !4
  %120 = getelementptr inbounds float, ptr %75, i32 6
  %121 = load float, ptr %120, align 4, !tbaa !4
  %122 = fpext float %121 to double
  %123 = load i64, ptr %45, align 4, !tbaa !6
  %124 = getelementptr inbounds double, ptr %39, i64 %123
  store double %122, ptr %124, align 8, !tbaa !1
  %125 = add i64 %123, 1
  store i64 %125, ptr %45, align 4, !tbaa !6
  %126 = fpext float %119 to double
  %127 = load i64, ptr %45, align 4, !tbaa !6
  %128 = getelementptr inbounds double, ptr %39, i64 %127
  store double %126, ptr %128, align 8, !tbaa !1
  %129 = add i64 %127, 1
  store i64 %129, ptr %45, align 4, !tbaa !6
  %130 = fpext float %117 to double
  %131 = load i64, ptr %45, align 4, !tbaa !6
  %132 = getelementptr inbounds double, ptr %39, i64 %131
  store double %130, ptr %132, align 8, !tbaa !1
  %133 = add i64 %131, 1
  store i64 %133, ptr %45, align 4, !tbaa !6
  %134 = fpext float %115 to double
  %135 = load i64, ptr %45, align 4, !tbaa !6
  %136 = getelementptr inbounds double, ptr %39, i64 %135
  store double %134, ptr %136, align 8, !tbaa !1
  %137 = add i64 %135, 1
  store i64 %137, ptr %45, align 4, !tbaa !6
  %138 = getelementptr inbounds float, ptr %1, i32 17
  %139 = load float, ptr %138, align 4, !tbaa !4
  %140 = getelementptr inbounds float, ptr %1, i32 16
  %141 = load float, ptr %140, align 4, !tbaa !4
  %142 = getelementptr inbounds float, ptr %1, i32 15
  %143 = load float, ptr %142, align 4, !tbaa !4
  %144 = getelementptr inbounds float, ptr %75, i32 5
  %145 = load float, ptr %144, align 4, !tbaa !4
  %146 = fpext float %145 to double
  %147 = load i64, ptr %45, align 4, !tbaa !6
  %148 = getelementptr inbounds double, ptr %39, i64 %147
  store double %146, ptr %148, align 8, !tbaa !1
  %149 = add i64 %147, 1
  store i64 %149, ptr %45, align 4, !tbaa !6
  %150 = fpext float %143 to double
  %151 = load i64, ptr %45, align 4, !tbaa !6
  %152 = getelementptr inbounds double, ptr %39, i64 %151
  store double %150, ptr %152, align 8, !tbaa !1
  %153 = add i64 %151, 1
  store i64 %153, ptr %45, align 4, !tbaa !6
  %154 = fpext float %141 to double
  %155 = load i64, ptr %45, align 4, !tbaa !6
  %156 = getelementptr inbounds double, ptr %39, i64 %155
  store double %154, ptr %156, align 8, !tbaa !1
  %157 = add i64 %155, 1
  store i64 %157, ptr %45, align 4, !tbaa !6
  %158 = fpext float %139 to double
  %159 = load i64, ptr %45, align 4, !tbaa !6
  %160 = getelementptr inbounds double, ptr %39, i64 %159
  store double %158, ptr %160, align 8, !tbaa !1
  %161 = add i64 %159, 1
  store i64 %161, ptr %45, align 4, !tbaa !6
  %162 = getelementptr inbounds float, ptr %1, i32 14
  %163 = load float, ptr %162, align 4, !tbaa !4
  %164 = getelementptr inbounds float, ptr %1, i32 13
  %165 = load float, ptr %164, align 4, !tbaa !4
  %166 = getelementptr inbounds float, ptr %1, i32 12
  %167 = load float, ptr %166, align 4, !tbaa !4
  %168 = getelementptr inbounds float, ptr %75, i32 4
  %169 = load float, ptr %168, align 4, !tbaa !4
  %170 = fpext float %169 to double
  %171 = load i64, ptr %45, align 4, !tbaa !6
  %172 = getelementptr inbounds double, ptr %39, i64 %171
  store double %170, ptr %172, align 8, !tbaa !1
  %173 = add i64 %171, 1
  store i64 %173, ptr %45, align 4, !tbaa !6
  %174 = fpext float %167 to double
  %175 = load i64, ptr %45, align 4, !tbaa !6
  %176 = getelementptr inbounds double, ptr %39, i64 %175
  store double %174, ptr %176, align 8, !tbaa !1
  %177 = add i64 %175, 1
  store i64 %177, ptr %45, align 4, !tbaa !6
  %178 = fpext float %165 to double
  %179 = load i64, ptr %45, align 4, !tbaa !6
  %180 = getelementptr inbounds double, ptr %39, i64 %179
  store double %178, ptr %180, align 8, !tbaa !1
  %181 = add i64 %179, 1
  store i64 %181, ptr %45, align 4, !tbaa !6
  %182 = fpext float %163 to double
  %183 = load i64, ptr %45, align 4, !tbaa !6
  %184 = getelementptr inbounds double, ptr %39, i64 %183
  store double %182, ptr %184, align 8, !tbaa !1
  %185 = add i64 %183, 1
  store i64 %185, ptr %45, align 4, !tbaa !6
  %186 = getelementptr inbounds float, ptr %1, i32 11
  %187 = load float, ptr %186, align 4, !tbaa !4
  %188 = getelementptr inbounds float, ptr %1, i32 10
  %189 = load float, ptr %188, align 4, !tbaa !4
  %190 = getelementptr inbounds float, ptr %1, i32 9
  %191 = load float, ptr %190, align 4, !tbaa !4
  %192 = getelementptr inbounds float, ptr %75, i32 3
  %193 = load float, ptr %192, align 4, !tbaa !4
  %194 = fpext float %193 to double
  %195 = load i64, ptr %45, align 4, !tbaa !6
  %196 = getelementptr inbounds double, ptr %39, i64 %195
  store double %194, ptr %196, align 8, !tbaa !1
  %197 = add i64 %195, 1
  store i64 %197, ptr %45, align 4, !tbaa !6
  %198 = fpext float %191 to double
  %199 = load i64, ptr %45, align 4, !tbaa !6
  %200 = getelementptr inbounds double, ptr %39, i64 %199
  store double %198, ptr %200, align 8, !tbaa !1
  %201 = add i64 %199, 1
  store i64 %201, ptr %45, align 4, !tbaa !6
  %202 = fpext float %189 to double
  %203 = load i64, ptr %45, align 4, !tbaa !6
  %204 = getelementptr inbounds double, ptr %39, i64 %203
  store double %202, ptr %204, align 8, !tbaa !1
  %205 = add i64 %203, 1
  store i64 %205, ptr %45, align 4, !tbaa !6
  %206 = fpext float %187 to double
  %207 = load i64, ptr %45, align 4, !tbaa !6
  %208 = getelementptr inbounds double, ptr %39, i64 %207
  store double %206, ptr %208, align 8, !tbaa !1
  %209 = add i64 %207, 1
  store i64 %209, ptr %45, align 4, !tbaa !6
  %210 = getelementptr inbounds float, ptr %1, i32 8
  %211 = load float, ptr %210, align 4, !tbaa !4
  %212 = getelementptr inbounds float, ptr %1, i32 7
  %213 = load float, ptr %212, align 4, !tbaa !4
  %214 = getelementptr inbounds float, ptr %1, i32 6
  %215 = load float, ptr %214, align 4, !tbaa !4
  %216 = getelementptr inbounds float, ptr %75, i32 2
  %217 = load float, ptr %216, align 4, !tbaa !4
  %218 = fpext float %217 to double
  %219 = load i64, ptr %45, align 4, !tbaa !6
  %220 = getelementptr inbounds double, ptr %39, i64 %219
  store double %218, ptr %220, align 8, !tbaa !1
  %221 = add i64 %219, 1
  store i64 %221, ptr %45, align 4, !tbaa !6
  %222 = fpext float %215 to double
  %223 = load i64, ptr %45, align 4, !tbaa !6
  %224 = getelementptr inbounds double, ptr %39, i64 %223
  store double %222, ptr %224, align 8, !tbaa !1
  %225 = add i64 %223, 1
  store i64 %225, ptr %45, align 4, !tbaa !6
  %226 = fpext float %213 to double
  %227 = load i64, ptr %45, align 4, !tbaa !6
  %228 = getelementptr inbounds double, ptr %39, i64 %227
  store double %226, ptr %228, align 8, !tbaa !1
  %229 = add i64 %227, 1
  store i64 %229, ptr %45, align 4, !tbaa !6
  %230 = fpext float %211 to double
  %231 = load i64, ptr %45, align 4, !tbaa !6
  %232 = getelementptr inbounds double, ptr %39, i64 %231
  store double %230, ptr %232, align 8, !tbaa !1
  %233 = add i64 %231, 1
  store i64 %233, ptr %45, align 4, !tbaa !6
  %234 = getelementptr inbounds float, ptr %1, i32 2
  %235 = load float, ptr %234, align 4, !tbaa !4
  %236 = getelementptr inbounds float, ptr %1, i32 1
  %237 = load float, ptr %236, align 4, !tbaa !4
  %238 = load float, ptr %1, align 4, !tbaa !4
  %239 = load float, ptr %75, align 4, !tbaa !4
  %240 = fpext float %239 to double
  %241 = load i64, ptr %45, align 4, !tbaa !6
  %242 = getelementptr inbounds double, ptr %39, i64 %241
  store double %240, ptr %242, align 8, !tbaa !1
  %243 = add i64 %241, 1
  store i64 %243, ptr %45, align 4, !tbaa !6
  %244 = fpext float %238 to double
  %245 = load i64, ptr %45, align 4, !tbaa !6
  %246 = getelementptr inbounds double, ptr %39, i64 %245
  store double %244, ptr %246, align 8, !tbaa !1
  %247 = add i64 %245, 1
  store i64 %247, ptr %45, align 4, !tbaa !6
  %248 = fpext float %237 to double
  %249 = load i64, ptr %45, align 4, !tbaa !6
  %250 = getelementptr inbounds double, ptr %39, i64 %249
  store double %248, ptr %250, align 8, !tbaa !1
  %251 = add i64 %249, 1
  store i64 %251, ptr %45, align 4, !tbaa !6
  %252 = fpext float %235 to double
  %253 = load i64, ptr %45, align 4, !tbaa !6
  %254 = getelementptr inbounds double, ptr %39, i64 %253
  store double %252, ptr %254, align 8, !tbaa !1
  %255 = add i64 %253, 1
  store i64 %255, ptr %45, align 4, !tbaa !6
  %256 = getelementptr inbounds float, ptr %1, i32 5
  %257 = load float, ptr %256, align 4, !tbaa !4
  %258 = getelementptr inbounds float, ptr %1, i32 4
  %259 = load float, ptr %258, align 4, !tbaa !4
  %260 = getelementptr inbounds float, ptr %1, i32 3
  %261 = load float, ptr %260, align 4, !tbaa !4
  %262 = getelementptr inbounds float, ptr %75, i32 1
  %263 = load float, ptr %262, align 4, !tbaa !4
  call void @_mlir_memref_to_llvm_free(ptr %70)
  %264 = fpext float %263 to double
  %265 = load i64, ptr %45, align 4, !tbaa !6
  %266 = getelementptr inbounds double, ptr %39, i64 %265
  store double %264, ptr %266, align 8, !tbaa !1
  %267 = add i64 %265, 1
  store i64 %267, ptr %45, align 4, !tbaa !6
  %268 = fpext float %261 to double
  %269 = load i64, ptr %45, align 4, !tbaa !6
  %270 = getelementptr inbounds double, ptr %39, i64 %269
  store double %268, ptr %270, align 8, !tbaa !1
  %271 = add i64 %269, 1
  store i64 %271, ptr %45, align 4, !tbaa !6
  %272 = fpext float %259 to double
  %273 = load i64, ptr %45, align 4, !tbaa !6
  %274 = getelementptr inbounds double, ptr %39, i64 %273
  store double %272, ptr %274, align 8, !tbaa !1
  %275 = add i64 %273, 1
  store i64 %275, ptr %45, align 4, !tbaa !6
  %276 = fpext float %257 to double
  %277 = load i64, ptr %45, align 4, !tbaa !6
  %278 = getelementptr inbounds double, ptr %39, i64 %277
  store double %276, ptr %278, align 8, !tbaa !1
  %279 = add i64 %277, 1
  store i64 %279, ptr %45, align 4, !tbaa !6
  %280 = fpext float %63 to double
  %281 = load i64, ptr %45, align 4, !tbaa !6
  %282 = getelementptr inbounds double, ptr %39, i64 %281
  store double %280, ptr %282, align 8, !tbaa !1
  %283 = add i64 %281, 1
  store i64 %283, ptr %45, align 4, !tbaa !6
  %284 = fpext float %61 to double
  %285 = load i64, ptr %45, align 4, !tbaa !6
  %286 = getelementptr inbounds double, ptr %39, i64 %285
  store double %284, ptr %286, align 8, !tbaa !1
  %287 = add i64 %285, 1
  store i64 %287, ptr %45, align 4, !tbaa !6
  %288 = fpext float %59 to double
  %289 = load i64, ptr %45, align 4, !tbaa !6
  %290 = getelementptr inbounds double, ptr %39, i64 %289
  store double %288, ptr %290, align 8, !tbaa !1
  %291 = add i64 %289, 1
  store i64 %291, ptr %45, align 4, !tbaa !6
  %292 = getelementptr inbounds float, ptr %1, i32 38
  %293 = load float, ptr %292, align 4, !tbaa !4
  %294 = getelementptr inbounds float, ptr %1, i32 37
  %295 = load float, ptr %294, align 4, !tbaa !4
  %296 = getelementptr inbounds float, ptr %1, i32 36
  %297 = load float, ptr %296, align 4, !tbaa !4
  %298 = fpext float %297 to double
  %299 = load i64, ptr %45, align 4, !tbaa !6
  %300 = getelementptr inbounds double, ptr %39, i64 %299
  store double %298, ptr %300, align 8, !tbaa !1
  %301 = add i64 %299, 1
  store i64 %301, ptr %45, align 4, !tbaa !6
  %302 = fpext float %295 to double
  %303 = load i64, ptr %45, align 4, !tbaa !6
  %304 = getelementptr inbounds double, ptr %39, i64 %303
  store double %302, ptr %304, align 8, !tbaa !1
  %305 = add i64 %303, 1
  store i64 %305, ptr %45, align 4, !tbaa !6
  %306 = fpext float %293 to double
  %307 = load i64, ptr %45, align 4, !tbaa !6
  %308 = getelementptr inbounds double, ptr %39, i64 %307
  store double %306, ptr %308, align 8, !tbaa !1
  %309 = add i64 %307, 1
  store i64 %309, ptr %45, align 4, !tbaa !6
  %310 = getelementptr inbounds float, ptr %1, i32 26
  %311 = load float, ptr %310, align 4, !tbaa !4
  %312 = getelementptr inbounds float, ptr %1, i32 25
  %313 = load float, ptr %312, align 4, !tbaa !4
  %314 = getelementptr inbounds float, ptr %1, i32 24
  %315 = load float, ptr %314, align 4, !tbaa !4
  %316 = fpext float %315 to double
  %317 = load i64, ptr %45, align 4, !tbaa !6
  %318 = getelementptr inbounds double, ptr %39, i64 %317
  store double %316, ptr %318, align 8, !tbaa !1
  %319 = add i64 %317, 1
  store i64 %319, ptr %45, align 4, !tbaa !6
  %320 = fpext float %313 to double
  %321 = load i64, ptr %45, align 4, !tbaa !6
  %322 = getelementptr inbounds double, ptr %39, i64 %321
  store double %320, ptr %322, align 8, !tbaa !1
  %323 = add i64 %321, 1
  store i64 %323, ptr %45, align 4, !tbaa !6
  %324 = fpext float %311 to double
  %325 = load i64, ptr %45, align 4, !tbaa !6
  %326 = getelementptr inbounds double, ptr %39, i64 %325
  store double %324, ptr %326, align 8, !tbaa !1
  %327 = add i64 %325, 1
  store i64 %327, ptr %45, align 4, !tbaa !6
  %328 = getelementptr inbounds float, ptr %1, i32 32
  %329 = load float, ptr %328, align 4, !tbaa !4
  %330 = getelementptr inbounds float, ptr %1, i32 31
  %331 = load float, ptr %330, align 4, !tbaa !4
  %332 = getelementptr inbounds float, ptr %1, i32 30
  %333 = load float, ptr %332, align 4, !tbaa !4
  %334 = fpext float %333 to double
  %335 = load i64, ptr %45, align 4, !tbaa !6
  %336 = getelementptr inbounds double, ptr %39, i64 %335
  store double %334, ptr %336, align 8, !tbaa !1
  %337 = add i64 %335, 1
  store i64 %337, ptr %45, align 4, !tbaa !6
  %338 = fpext float %331 to double
  %339 = load i64, ptr %45, align 4, !tbaa !6
  %340 = getelementptr inbounds double, ptr %39, i64 %339
  store double %338, ptr %340, align 8, !tbaa !1
  %341 = add i64 %339, 1
  store i64 %341, ptr %45, align 4, !tbaa !6
  %342 = fpext float %329 to double
  %343 = load i64, ptr %45, align 4, !tbaa !6
  %344 = getelementptr inbounds double, ptr %39, i64 %343
  store double %342, ptr %344, align 8, !tbaa !1
  %345 = add i64 %343, 1
  store i64 %345, ptr %45, align 4, !tbaa !6
  %346 = fpext float %57 to double
  %347 = load i64, ptr %45, align 4, !tbaa !6
  %348 = getelementptr inbounds double, ptr %39, i64 %347
  store double %346, ptr %348, align 8, !tbaa !1
  %349 = add i64 %347, 1
  store i64 %349, ptr %45, align 4, !tbaa !6
  %350 = fpext float %55 to double
  %351 = load i64, ptr %45, align 4, !tbaa !6
  %352 = getelementptr inbounds double, ptr %39, i64 %351
  store double %350, ptr %352, align 8, !tbaa !1
  %353 = add i64 %351, 1
  store i64 %353, ptr %45, align 4, !tbaa !6
  %354 = fpext float %53 to double
  %355 = load i64, ptr %45, align 4, !tbaa !6
  %356 = getelementptr inbounds double, ptr %39, i64 %355
  store double %354, ptr %356, align 8, !tbaa !1
  %357 = add i64 %355, 1
  store i64 %357, ptr %45, align 4, !tbaa !6
  %358 = getelementptr inbounds float, ptr %1, i32 59
  %359 = load float, ptr %358, align 4, !tbaa !4
  %360 = getelementptr inbounds float, ptr %1, i32 58
  %361 = load float, ptr %360, align 4, !tbaa !4
  %362 = getelementptr inbounds float, ptr %1, i32 57
  %363 = load float, ptr %362, align 4, !tbaa !4
  %364 = getelementptr inbounds float, ptr %1, i32 41
  %365 = load float, ptr %364, align 4, !tbaa !4
  %366 = getelementptr inbounds float, ptr %1, i32 40
  %367 = load float, ptr %366, align 4, !tbaa !4
  %368 = getelementptr inbounds float, ptr %1, i32 39
  %369 = load float, ptr %368, align 4, !tbaa !4
  %370 = fpext float %369 to double
  %371 = load i64, ptr %45, align 4, !tbaa !6
  %372 = getelementptr inbounds double, ptr %39, i64 %371
  store double %370, ptr %372, align 8, !tbaa !1
  %373 = add i64 %371, 1
  store i64 %373, ptr %45, align 4, !tbaa !6
  %374 = fpext float %367 to double
  %375 = load i64, ptr %45, align 4, !tbaa !6
  %376 = getelementptr inbounds double, ptr %39, i64 %375
  store double %374, ptr %376, align 8, !tbaa !1
  %377 = add i64 %375, 1
  store i64 %377, ptr %45, align 4, !tbaa !6
  %378 = fpext float %365 to double
  %379 = load i64, ptr %45, align 4, !tbaa !6
  %380 = getelementptr inbounds double, ptr %39, i64 %379
  store double %378, ptr %380, align 8, !tbaa !1
  %381 = add i64 %379, 1
  store i64 %381, ptr %45, align 4, !tbaa !6
  %382 = getelementptr inbounds float, ptr %1, i32 29
  %383 = load float, ptr %382, align 4, !tbaa !4
  %384 = getelementptr inbounds float, ptr %1, i32 28
  %385 = load float, ptr %384, align 4, !tbaa !4
  %386 = getelementptr inbounds float, ptr %1, i32 27
  %387 = load float, ptr %386, align 4, !tbaa !4
  %388 = fpext float %387 to double
  %389 = load i64, ptr %45, align 4, !tbaa !6
  %390 = getelementptr inbounds double, ptr %39, i64 %389
  store double %388, ptr %390, align 8, !tbaa !1
  %391 = add i64 %389, 1
  store i64 %391, ptr %45, align 4, !tbaa !6
  %392 = fpext float %385 to double
  %393 = load i64, ptr %45, align 4, !tbaa !6
  %394 = getelementptr inbounds double, ptr %39, i64 %393
  store double %392, ptr %394, align 8, !tbaa !1
  %395 = add i64 %393, 1
  store i64 %395, ptr %45, align 4, !tbaa !6
  %396 = fpext float %383 to double
  %397 = load i64, ptr %45, align 4, !tbaa !6
  %398 = getelementptr inbounds double, ptr %39, i64 %397
  store double %396, ptr %398, align 8, !tbaa !1
  %399 = add i64 %397, 1
  store i64 %399, ptr %45, align 4, !tbaa !6
  %400 = getelementptr inbounds float, ptr %1, i32 35
  %401 = load float, ptr %400, align 4, !tbaa !4
  %402 = getelementptr inbounds float, ptr %1, i32 34
  %403 = load float, ptr %402, align 4, !tbaa !4
  %404 = getelementptr inbounds float, ptr %1, i32 33
  %405 = load float, ptr %404, align 4, !tbaa !4
  %406 = fpext float %405 to double
  %407 = load i64, ptr %45, align 4, !tbaa !6
  %408 = getelementptr inbounds double, ptr %39, i64 %407
  store double %406, ptr %408, align 8, !tbaa !1
  %409 = add i64 %407, 1
  store i64 %409, ptr %45, align 4, !tbaa !6
  %410 = fpext float %403 to double
  %411 = load i64, ptr %45, align 4, !tbaa !6
  %412 = getelementptr inbounds double, ptr %39, i64 %411
  store double %410, ptr %412, align 8, !tbaa !1
  %413 = add i64 %411, 1
  store i64 %413, ptr %45, align 4, !tbaa !6
  %414 = fpext float %401 to double
  %415 = load i64, ptr %45, align 4, !tbaa !6
  %416 = getelementptr inbounds double, ptr %39, i64 %415
  store double %414, ptr %416, align 8, !tbaa !1
  %417 = add i64 %415, 1
  store i64 %417, ptr %45, align 4, !tbaa !6
  %418 = fpext float %363 to double
  %419 = load i64, ptr %45, align 4, !tbaa !6
  %420 = getelementptr inbounds double, ptr %39, i64 %419
  store double %418, ptr %420, align 8, !tbaa !1
  %421 = add i64 %419, 1
  store i64 %421, ptr %45, align 4, !tbaa !6
  %422 = fpext float %361 to double
  %423 = load i64, ptr %45, align 4, !tbaa !6
  %424 = getelementptr inbounds double, ptr %39, i64 %423
  store double %422, ptr %424, align 8, !tbaa !1
  %425 = add i64 %423, 1
  store i64 %425, ptr %45, align 4, !tbaa !6
  %426 = fpext float %359 to double
  %427 = load i64, ptr %45, align 4, !tbaa !6
  %428 = getelementptr inbounds double, ptr %39, i64 %427
  store double %426, ptr %428, align 8, !tbaa !1
  %429 = add i64 %427, 1
  store i64 %429, ptr %45, align 4, !tbaa !6
  %430 = getelementptr inbounds float, ptr %1, i32 56
  %431 = load float, ptr %430, align 4, !tbaa !4
  %432 = getelementptr inbounds float, ptr %1, i32 55
  %433 = load float, ptr %432, align 4, !tbaa !4
  %434 = getelementptr inbounds float, ptr %1, i32 54
  %435 = load float, ptr %434, align 4, !tbaa !4
  %436 = fpext float %435 to double
  %437 = load i64, ptr %45, align 4, !tbaa !6
  %438 = getelementptr inbounds double, ptr %39, i64 %437
  store double %436, ptr %438, align 8, !tbaa !1
  %439 = add i64 %437, 1
  store i64 %439, ptr %45, align 4, !tbaa !6
  %440 = fpext float %433 to double
  %441 = load i64, ptr %45, align 4, !tbaa !6
  %442 = getelementptr inbounds double, ptr %39, i64 %441
  store double %440, ptr %442, align 8, !tbaa !1
  %443 = add i64 %441, 1
  store i64 %443, ptr %45, align 4, !tbaa !6
  %444 = fpext float %431 to double
  %445 = load i64, ptr %45, align 4, !tbaa !6
  %446 = getelementptr inbounds double, ptr %39, i64 %445
  store double %444, ptr %446, align 8, !tbaa !1
  %447 = add i64 %445, 1
  store i64 %447, ptr %45, align 4, !tbaa !6
  %448 = getelementptr inbounds float, ptr %1, i32 65
  %449 = load float, ptr %448, align 4, !tbaa !4
  %450 = getelementptr inbounds float, ptr %1, i32 64
  %451 = load float, ptr %450, align 4, !tbaa !4
  %452 = getelementptr inbounds float, ptr %1, i32 63
  %453 = load float, ptr %452, align 4, !tbaa !4
  %454 = getelementptr inbounds float, ptr %1, i32 47
  %455 = load float, ptr %454, align 4, !tbaa !4
  %456 = getelementptr inbounds float, ptr %1, i32 46
  %457 = load float, ptr %456, align 4, !tbaa !4
  %458 = getelementptr inbounds float, ptr %1, i32 45
  %459 = load float, ptr %458, align 4, !tbaa !4
  %460 = fpext float %459 to double
  %461 = load i64, ptr %45, align 4, !tbaa !6
  %462 = getelementptr inbounds double, ptr %39, i64 %461
  store double %460, ptr %462, align 8, !tbaa !1
  %463 = add i64 %461, 1
  store i64 %463, ptr %45, align 4, !tbaa !6
  %464 = fpext float %457 to double
  %465 = load i64, ptr %45, align 4, !tbaa !6
  %466 = getelementptr inbounds double, ptr %39, i64 %465
  store double %464, ptr %466, align 8, !tbaa !1
  %467 = add i64 %465, 1
  store i64 %467, ptr %45, align 4, !tbaa !6
  %468 = fpext float %455 to double
  %469 = load i64, ptr %45, align 4, !tbaa !6
  %470 = getelementptr inbounds double, ptr %39, i64 %469
  store double %468, ptr %470, align 8, !tbaa !1
  %471 = add i64 %469, 1
  store i64 %471, ptr %45, align 4, !tbaa !6
  %472 = fpext float %453 to double
  %473 = load i64, ptr %45, align 4, !tbaa !6
  %474 = getelementptr inbounds double, ptr %39, i64 %473
  store double %472, ptr %474, align 8, !tbaa !1
  %475 = add i64 %473, 1
  store i64 %475, ptr %45, align 4, !tbaa !6
  %476 = fpext float %451 to double
  %477 = load i64, ptr %45, align 4, !tbaa !6
  %478 = getelementptr inbounds double, ptr %39, i64 %477
  store double %476, ptr %478, align 8, !tbaa !1
  %479 = add i64 %477, 1
  store i64 %479, ptr %45, align 4, !tbaa !6
  %480 = fpext float %449 to double
  %481 = load i64, ptr %45, align 4, !tbaa !6
  %482 = getelementptr inbounds double, ptr %39, i64 %481
  store double %480, ptr %482, align 8, !tbaa !1
  %483 = add i64 %481, 1
  store i64 %483, ptr %45, align 4, !tbaa !6
  %484 = fpext float %51 to double
  %485 = load i64, ptr %45, align 4, !tbaa !6
  %486 = getelementptr inbounds double, ptr %39, i64 %485
  store double %484, ptr %486, align 8, !tbaa !1
  %487 = add i64 %485, 1
  store i64 %487, ptr %45, align 4, !tbaa !6
  %488 = fpext float %49 to double
  %489 = load i64, ptr %45, align 4, !tbaa !6
  %490 = getelementptr inbounds double, ptr %39, i64 %489
  store double %488, ptr %490, align 8, !tbaa !1
  %491 = add i64 %489, 1
  store i64 %491, ptr %45, align 4, !tbaa !6
  %492 = fpext float %47 to double
  %493 = load i64, ptr %45, align 4, !tbaa !6
  %494 = getelementptr inbounds double, ptr %39, i64 %493
  store double %492, ptr %494, align 8, !tbaa !1
  %495 = add i64 %493, 1
  store i64 %495, ptr %45, align 4, !tbaa !6
  %496 = getelementptr inbounds float, ptr %1, i32 86
  %497 = load float, ptr %496, align 4, !tbaa !4
  %498 = getelementptr inbounds float, ptr %1, i32 85
  %499 = load float, ptr %498, align 4, !tbaa !4
  %500 = getelementptr inbounds float, ptr %1, i32 84
  %501 = load float, ptr %500, align 4, !tbaa !4
  %502 = getelementptr inbounds float, ptr %1, i32 71
  %503 = load float, ptr %502, align 4, !tbaa !4
  %504 = getelementptr inbounds float, ptr %1, i32 70
  %505 = load float, ptr %504, align 4, !tbaa !4
  %506 = getelementptr inbounds float, ptr %1, i32 69
  %507 = load float, ptr %506, align 4, !tbaa !4
  %508 = fpext float %507 to double
  %509 = load i64, ptr %45, align 4, !tbaa !6
  %510 = getelementptr inbounds double, ptr %39, i64 %509
  store double %508, ptr %510, align 8, !tbaa !1
  %511 = add i64 %509, 1
  store i64 %511, ptr %45, align 4, !tbaa !6
  %512 = fpext float %505 to double
  %513 = load i64, ptr %45, align 4, !tbaa !6
  %514 = getelementptr inbounds double, ptr %39, i64 %513
  store double %512, ptr %514, align 8, !tbaa !1
  %515 = add i64 %513, 1
  store i64 %515, ptr %45, align 4, !tbaa !6
  %516 = fpext float %503 to double
  %517 = load i64, ptr %45, align 4, !tbaa !6
  %518 = getelementptr inbounds double, ptr %39, i64 %517
  store double %516, ptr %518, align 8, !tbaa !1
  %519 = add i64 %517, 1
  store i64 %519, ptr %45, align 4, !tbaa !6
  %520 = getelementptr inbounds float, ptr %1, i32 53
  %521 = load float, ptr %520, align 4, !tbaa !4
  %522 = getelementptr inbounds float, ptr %1, i32 52
  %523 = load float, ptr %522, align 4, !tbaa !4
  %524 = getelementptr inbounds float, ptr %1, i32 51
  %525 = load float, ptr %524, align 4, !tbaa !4
  %526 = fpext float %525 to double
  %527 = load i64, ptr %45, align 4, !tbaa !6
  %528 = getelementptr inbounds double, ptr %39, i64 %527
  store double %526, ptr %528, align 8, !tbaa !1
  %529 = add i64 %527, 1
  store i64 %529, ptr %45, align 4, !tbaa !6
  %530 = fpext float %523 to double
  %531 = load i64, ptr %45, align 4, !tbaa !6
  %532 = getelementptr inbounds double, ptr %39, i64 %531
  store double %530, ptr %532, align 8, !tbaa !1
  %533 = add i64 %531, 1
  store i64 %533, ptr %45, align 4, !tbaa !6
  %534 = fpext float %521 to double
  %535 = load i64, ptr %45, align 4, !tbaa !6
  %536 = getelementptr inbounds double, ptr %39, i64 %535
  store double %534, ptr %536, align 8, !tbaa !1
  %537 = add i64 %535, 1
  store i64 %537, ptr %45, align 4, !tbaa !6
  %538 = getelementptr inbounds float, ptr %1, i32 62
  %539 = load float, ptr %538, align 4, !tbaa !4
  %540 = getelementptr inbounds float, ptr %1, i32 61
  %541 = load float, ptr %540, align 4, !tbaa !4
  %542 = getelementptr inbounds float, ptr %1, i32 60
  %543 = load float, ptr %542, align 4, !tbaa !4
  %544 = fpext float %543 to double
  %545 = load i64, ptr %45, align 4, !tbaa !6
  %546 = getelementptr inbounds double, ptr %39, i64 %545
  store double %544, ptr %546, align 8, !tbaa !1
  %547 = add i64 %545, 1
  store i64 %547, ptr %45, align 4, !tbaa !6
  %548 = fpext float %541 to double
  %549 = load i64, ptr %45, align 4, !tbaa !6
  %550 = getelementptr inbounds double, ptr %39, i64 %549
  store double %548, ptr %550, align 8, !tbaa !1
  %551 = add i64 %549, 1
  store i64 %551, ptr %45, align 4, !tbaa !6
  %552 = fpext float %539 to double
  %553 = load i64, ptr %45, align 4, !tbaa !6
  %554 = getelementptr inbounds double, ptr %39, i64 %553
  store double %552, ptr %554, align 8, !tbaa !1
  %555 = add i64 %553, 1
  store i64 %555, ptr %45, align 4, !tbaa !6
  %556 = fpext float %501 to double
  %557 = load i64, ptr %45, align 4, !tbaa !6
  %558 = getelementptr inbounds double, ptr %39, i64 %557
  store double %556, ptr %558, align 8, !tbaa !1
  %559 = add i64 %557, 1
  store i64 %559, ptr %45, align 4, !tbaa !6
  %560 = fpext float %499 to double
  %561 = load i64, ptr %45, align 4, !tbaa !6
  %562 = getelementptr inbounds double, ptr %39, i64 %561
  store double %560, ptr %562, align 8, !tbaa !1
  %563 = add i64 %561, 1
  store i64 %563, ptr %45, align 4, !tbaa !6
  %564 = fpext float %497 to double
  %565 = load i64, ptr %45, align 4, !tbaa !6
  %566 = getelementptr inbounds double, ptr %39, i64 %565
  store double %564, ptr %566, align 8, !tbaa !1
  %567 = add i64 %565, 1
  store i64 %567, ptr %45, align 4, !tbaa !6
  %568 = getelementptr inbounds float, ptr %1, i32 77
  %569 = load float, ptr %568, align 4, !tbaa !4
  %570 = getelementptr inbounds float, ptr %1, i32 76
  %571 = load float, ptr %570, align 4, !tbaa !4
  %572 = getelementptr inbounds float, ptr %1, i32 75
  %573 = load float, ptr %572, align 4, !tbaa !4
  %574 = getelementptr inbounds float, ptr %1, i32 68
  %575 = load float, ptr %574, align 4, !tbaa !4
  %576 = getelementptr inbounds float, ptr %1, i32 67
  %577 = load float, ptr %576, align 4, !tbaa !4
  %578 = getelementptr inbounds float, ptr %1, i32 66
  %579 = load float, ptr %578, align 4, !tbaa !4
  %580 = fpext float %579 to double
  %581 = load i64, ptr %45, align 4, !tbaa !6
  %582 = getelementptr inbounds double, ptr %39, i64 %581
  store double %580, ptr %582, align 8, !tbaa !1
  %583 = add i64 %581, 1
  store i64 %583, ptr %45, align 4, !tbaa !6
  %584 = fpext float %577 to double
  %585 = load i64, ptr %45, align 4, !tbaa !6
  %586 = getelementptr inbounds double, ptr %39, i64 %585
  store double %584, ptr %586, align 8, !tbaa !1
  %587 = add i64 %585, 1
  store i64 %587, ptr %45, align 4, !tbaa !6
  %588 = fpext float %575 to double
  %589 = load i64, ptr %45, align 4, !tbaa !6
  %590 = getelementptr inbounds double, ptr %39, i64 %589
  store double %588, ptr %590, align 8, !tbaa !1
  %591 = add i64 %589, 1
  store i64 %591, ptr %45, align 4, !tbaa !6
  %592 = fpext float %573 to double
  %593 = load i64, ptr %45, align 4, !tbaa !6
  %594 = getelementptr inbounds double, ptr %39, i64 %593
  store double %592, ptr %594, align 8, !tbaa !1
  %595 = add i64 %593, 1
  store i64 %595, ptr %45, align 4, !tbaa !6
  %596 = fpext float %571 to double
  %597 = load i64, ptr %45, align 4, !tbaa !6
  %598 = getelementptr inbounds double, ptr %39, i64 %597
  store double %596, ptr %598, align 8, !tbaa !1
  %599 = add i64 %597, 1
  store i64 %599, ptr %45, align 4, !tbaa !6
  %600 = fpext float %569 to double
  %601 = load i64, ptr %45, align 4, !tbaa !6
  %602 = getelementptr inbounds double, ptr %39, i64 %601
  store double %600, ptr %602, align 8, !tbaa !1
  %603 = add i64 %601, 1
  store i64 %603, ptr %45, align 4, !tbaa !6
  %604 = getelementptr inbounds float, ptr %1, i32 89
  %605 = load float, ptr %604, align 4, !tbaa !4
  %606 = getelementptr inbounds float, ptr %1, i32 88
  %607 = load float, ptr %606, align 4, !tbaa !4
  %608 = getelementptr inbounds float, ptr %1, i32 87
  %609 = load float, ptr %608, align 4, !tbaa !4
  %610 = fpext float %609 to double
  %611 = load i64, ptr %45, align 4, !tbaa !6
  %612 = getelementptr inbounds double, ptr %39, i64 %611
  store double %610, ptr %612, align 8, !tbaa !1
  %613 = add i64 %611, 1
  store i64 %613, ptr %45, align 4, !tbaa !6
  %614 = fpext float %607 to double
  %615 = load i64, ptr %45, align 4, !tbaa !6
  %616 = getelementptr inbounds double, ptr %39, i64 %615
  store double %614, ptr %616, align 8, !tbaa !1
  %617 = add i64 %615, 1
  store i64 %617, ptr %45, align 4, !tbaa !6
  %618 = fpext float %605 to double
  %619 = load i64, ptr %45, align 4, !tbaa !6
  %620 = getelementptr inbounds double, ptr %39, i64 %619
  store double %618, ptr %620, align 8, !tbaa !1
  %621 = add i64 %619, 1
  store i64 %621, ptr %45, align 4, !tbaa !6
  %622 = getelementptr inbounds float, ptr %1, i32 80
  %623 = load float, ptr %622, align 4, !tbaa !4
  %624 = getelementptr inbounds float, ptr %1, i32 79
  %625 = load float, ptr %624, align 4, !tbaa !4
  %626 = getelementptr inbounds float, ptr %1, i32 78
  %627 = load float, ptr %626, align 4, !tbaa !4
  %628 = fpext float %627 to double
  %629 = load i64, ptr %45, align 4, !tbaa !6
  %630 = getelementptr inbounds double, ptr %39, i64 %629
  store double %628, ptr %630, align 8, !tbaa !1
  %631 = add i64 %629, 1
  store i64 %631, ptr %45, align 4, !tbaa !6
  %632 = fpext float %625 to double
  %633 = load i64, ptr %45, align 4, !tbaa !6
  %634 = getelementptr inbounds double, ptr %39, i64 %633
  store double %632, ptr %634, align 8, !tbaa !1
  %635 = add i64 %633, 1
  store i64 %635, ptr %45, align 4, !tbaa !6
  %636 = fpext float %623 to double
  %637 = load i64, ptr %45, align 4, !tbaa !6
  %638 = getelementptr inbounds double, ptr %39, i64 %637
  store double %636, ptr %638, align 8, !tbaa !1
  %639 = add i64 %637, 1
  store i64 %639, ptr %45, align 4, !tbaa !6
  %640 = getelementptr inbounds float, ptr %1, i32 92
  %641 = load float, ptr %640, align 4, !tbaa !4
  %642 = getelementptr inbounds float, ptr %1, i32 91
  %643 = load float, ptr %642, align 4, !tbaa !4
  %644 = getelementptr inbounds float, ptr %1, i32 90
  %645 = load float, ptr %644, align 4, !tbaa !4
  %646 = fpext float %645 to double
  %647 = load i64, ptr %45, align 4, !tbaa !6
  %648 = getelementptr inbounds double, ptr %39, i64 %647
  store double %646, ptr %648, align 8, !tbaa !1
  %649 = add i64 %647, 1
  store i64 %649, ptr %45, align 4, !tbaa !6
  %650 = fpext float %643 to double
  %651 = load i64, ptr %45, align 4, !tbaa !6
  %652 = getelementptr inbounds double, ptr %39, i64 %651
  store double %650, ptr %652, align 8, !tbaa !1
  %653 = add i64 %651, 1
  store i64 %653, ptr %45, align 4, !tbaa !6
  %654 = fpext float %641 to double
  %655 = load i64, ptr %45, align 4, !tbaa !6
  %656 = getelementptr inbounds double, ptr %39, i64 %655
  store double %654, ptr %656, align 8, !tbaa !1
  %657 = add i64 %655, 1
  store i64 %657, ptr %45, align 4, !tbaa !6
  %658 = getelementptr inbounds float, ptr %1, i32 83
  %659 = load float, ptr %658, align 4, !tbaa !4
  %660 = getelementptr inbounds float, ptr %1, i32 82
  %661 = load float, ptr %660, align 4, !tbaa !4
  %662 = getelementptr inbounds float, ptr %1, i32 81
  %663 = load float, ptr %662, align 4, !tbaa !4
  %664 = fpext float %663 to double
  %665 = load i64, ptr %45, align 4, !tbaa !6
  %666 = getelementptr inbounds double, ptr %39, i64 %665
  store double %664, ptr %666, align 8, !tbaa !1
  %667 = add i64 %665, 1
  store i64 %667, ptr %45, align 4, !tbaa !6
  %668 = fpext float %661 to double
  %669 = load i64, ptr %45, align 4, !tbaa !6
  %670 = getelementptr inbounds double, ptr %39, i64 %669
  store double %668, ptr %670, align 8, !tbaa !1
  %671 = add i64 %669, 1
  store i64 %671, ptr %45, align 4, !tbaa !6
  %672 = fpext float %659 to double
  %673 = load i64, ptr %45, align 4, !tbaa !6
  %674 = getelementptr inbounds double, ptr %39, i64 %673
  store double %672, ptr %674, align 8, !tbaa !1
  %675 = add i64 %673, 1
  store i64 %675, ptr %45, align 4, !tbaa !6
  %676 = getelementptr inbounds float, ptr %1, i32 95
  %677 = load float, ptr %676, align 4, !tbaa !4
  %678 = getelementptr inbounds float, ptr %1, i32 94
  %679 = load float, ptr %678, align 4, !tbaa !4
  %680 = getelementptr inbounds float, ptr %1, i32 93
  %681 = load float, ptr %680, align 4, !tbaa !4
  %682 = fpext float %681 to double
  %683 = load i64, ptr %45, align 4, !tbaa !6
  %684 = getelementptr inbounds double, ptr %39, i64 %683
  store double %682, ptr %684, align 8, !tbaa !1
  %685 = add i64 %683, 1
  store i64 %685, ptr %45, align 4, !tbaa !6
  %686 = fpext float %679 to double
  %687 = load i64, ptr %45, align 4, !tbaa !6
  %688 = getelementptr inbounds double, ptr %39, i64 %687
  store double %686, ptr %688, align 8, !tbaa !1
  %689 = add i64 %687, 1
  store i64 %689, ptr %45, align 4, !tbaa !6
  %690 = fpext float %677 to double
  %691 = load i64, ptr %45, align 4, !tbaa !6
  %692 = getelementptr inbounds double, ptr %39, i64 %691
  store double %690, ptr %692, align 8, !tbaa !1
  %693 = add i64 %691, 1
  store i64 %693, ptr %45, align 4, !tbaa !6
  store { ptr, ptr, i64, [3 x i64], [3 x i64] } %32, ptr %36, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %23, ptr %35, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %44, ptr %34, align 8
  %694 = call ptr @_mlir_memref_to_llvm_alloc(i64 8)
  %695 = insertvalue { ptr, ptr, i64 } poison, ptr %694, 0
  %696 = insertvalue { ptr, ptr, i64 } %695, ptr %694, 1
  %697 = insertvalue { ptr, ptr, i64 } %696, i64 0, 2
  store { ptr, ptr, i64 } %697, ptr %33, align 8
  call void @qnode_forward_0.quantum(ptr %36, ptr %35, ptr %34, ptr %33)
  %698 = load double, ptr %694, align 8, !tbaa !1
  store double %698, ptr %16, align 8, !tbaa !1
  ret void
}

define void @setup() {
  call void @__catalyst__rt__initialize(ptr null)
  ret void
}

define void @teardown() {
  call void @__catalyst__rt__finalize()
  ret void
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #1

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #2

attributes #0 = { noinline }
attributes #1 = { nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #2 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{!2, !2, i64 0}
!2 = !{!"double", !3, i64 0}
!3 = !{!"Catalyst TBAA"}
!4 = !{!5, !5, i64 0}
!5 = !{!"float", !3, i64 0}
!6 = !{!7, !7, i64 0}
!7 = !{!"int", !3, i64 0}
