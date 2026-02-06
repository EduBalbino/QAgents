; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

@"{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}" = internal constant [54 x i8] c"{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}\00"
@LightningSimulator = internal constant [19 x i8] c"LightningSimulator\00"
@"/home/pichau/QAgents/.venv/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.so" = internal constant [107 x i8] c"/home/pichau/QAgents/.venv/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.so\00"
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

declare void @_mlir_memref_to_llvm_free(ptr)

declare ptr @_mlir_memref_to_llvm_alloc(i64)

define { ptr, ptr, i64 } @jit_qnode_forward(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, ptr %9, ptr %10, i64 %11, i64 %12, i64 %13) {
  %15 = call { ptr, ptr, i64 } @qnode_forward_0(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, ptr %9, ptr %10, i64 %11, i64 %12, i64 %13)
  %16 = extractvalue { ptr, ptr, i64 } %15, 0
  %17 = ptrtoint ptr %16 to i64
  %18 = icmp eq i64 3735928559, %17
  br i1 %18, label %19, label %27

19:                                               ; preds = %14
  %20 = call ptr @_mlir_memref_to_llvm_alloc(i64 8)
  %21 = insertvalue { ptr, ptr, i64 } poison, ptr %20, 0
  %22 = insertvalue { ptr, ptr, i64 } %21, ptr %20, 1
  %23 = insertvalue { ptr, ptr, i64 } %22, i64 0, 2
  %24 = extractvalue { ptr, ptr, i64 } %15, 1
  %25 = extractvalue { ptr, ptr, i64 } %15, 2
  %26 = getelementptr inbounds double, ptr %24, i64 %25
  call void @llvm.memcpy.p0.p0.i64(ptr %20, ptr %26, i64 8, i1 false)
  br label %28

27:                                               ; preds = %14
  br label %28

28:                                               ; preds = %19, %27
  %29 = phi { ptr, ptr, i64 } [ %15, %27 ], [ %23, %19 ]
  br label %30

30:                                               ; preds = %28
  ret { ptr, ptr, i64 } %29
}

define void @_catalyst_pyface_jit_qnode_forward(ptr %0, ptr %1) {
  %3 = load { ptr, ptr, ptr }, ptr %1, align 8
  %4 = extractvalue { ptr, ptr, ptr } %3, 0
  %5 = extractvalue { ptr, ptr, ptr } %3, 1
  call void @_catalyst_ciface_jit_qnode_forward(ptr %0, ptr %4, ptr %5)
  ret void
}

define void @_catalyst_ciface_jit_qnode_forward(ptr %0, ptr %1, ptr %2) {
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
  %20 = call { ptr, ptr, i64 } @jit_qnode_forward(ptr %5, ptr %6, i64 %7, i64 %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13, ptr %15, ptr %16, i64 %17, i64 %18, i64 %19)
  store { ptr, ptr, i64 } %20, ptr %0, align 8
  ret void
}

define internal { ptr, ptr, i64 } @qnode_forward_0(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, ptr %9, ptr %10, i64 %11, i64 %12, i64 %13) {
  %15 = getelementptr inbounds float, ptr %1, i32 14
  %16 = load float, ptr %15, align 4
  %17 = getelementptr inbounds float, ptr %1, i32 13
  %18 = load float, ptr %17, align 4
  %19 = getelementptr inbounds float, ptr %1, i32 12
  %20 = load float, ptr %19, align 4
  %21 = getelementptr inbounds float, ptr %1, i32 11
  %22 = load float, ptr %21, align 4
  %23 = getelementptr inbounds float, ptr %1, i32 10
  %24 = load float, ptr %23, align 4
  %25 = getelementptr inbounds float, ptr %1, i32 9
  %26 = load float, ptr %25, align 4
  %27 = call ptr @_mlir_memref_to_llvm_alloc(i64 80)
  %28 = ptrtoint ptr %27 to i64
  %29 = add i64 %28, 63
  %30 = urem i64 %29, 64
  %31 = sub i64 %29, %30
  %32 = inttoptr i64 %31 to ptr
  br label %33

33:                                               ; preds = %36, %14
  %34 = phi i64 [ %39, %36 ], [ 0, %14 ]
  %35 = icmp slt i64 %34, 4
  br i1 %35, label %36, label %40

36:                                               ; preds = %33
  %37 = load float, ptr @__constant_xf32, align 4
  %38 = getelementptr inbounds float, ptr %32, i64 %34
  store float %37, ptr %38, align 4
  %39 = add i64 %34, 1
  br label %33

40:                                               ; preds = %33
  br label %41

41:                                               ; preds = %44, %40
  %42 = phi i64 [ %51, %44 ], [ 0, %40 ]
  %43 = icmp slt i64 %42, 4
  br i1 %43, label %44, label %52

44:                                               ; preds = %41
  %45 = getelementptr inbounds float, ptr %32, i64 %42
  %46 = load float, ptr %45, align 4
  %47 = getelementptr inbounds float, ptr %10, i64 %42
  %48 = load float, ptr %47, align 4
  %49 = fmul float %46, %48
  %50 = getelementptr inbounds float, ptr %32, i64 %42
  store float %49, ptr %50, align 4
  %51 = add i64 %42, 1
  br label %41

52:                                               ; preds = %41
  %53 = getelementptr inbounds float, ptr %32, i32 3
  %54 = load float, ptr %53, align 4
  call void @__catalyst__rt__device_init(ptr @"/home/pichau/QAgents/.venv/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.so", ptr @LightningSimulator, ptr @"{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}", i64 0, i1 false)
  %55 = call ptr @__catalyst__rt__qubit_allocate_array(i64 4)
  %56 = call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %55, i64 3)
  %57 = load ptr, ptr %56, align 8
  %58 = fpext float %54 to double
  call void @__catalyst__qis__RY(double %58, ptr %57, ptr null)
  %59 = fpext float %26 to double
  call void @__catalyst__qis__RZ(double %59, ptr %57, ptr null)
  %60 = fpext float %24 to double
  call void @__catalyst__qis__RY(double %60, ptr %57, ptr null)
  %61 = fpext float %22 to double
  call void @__catalyst__qis__RZ(double %61, ptr %57, ptr null)
  %62 = getelementptr inbounds float, ptr %1, i32 8
  %63 = load float, ptr %62, align 4
  %64 = getelementptr inbounds float, ptr %1, i32 7
  %65 = load float, ptr %64, align 4
  %66 = getelementptr inbounds float, ptr %1, i32 6
  %67 = load float, ptr %66, align 4
  %68 = getelementptr inbounds float, ptr %32, i32 2
  %69 = load float, ptr %68, align 4
  %70 = call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %55, i64 2)
  %71 = load ptr, ptr %70, align 8
  %72 = fpext float %69 to double
  call void @__catalyst__qis__RY(double %72, ptr %71, ptr null)
  %73 = fpext float %67 to double
  call void @__catalyst__qis__RZ(double %73, ptr %71, ptr null)
  %74 = fpext float %65 to double
  call void @__catalyst__qis__RY(double %74, ptr %71, ptr null)
  %75 = fpext float %63 to double
  call void @__catalyst__qis__RZ(double %75, ptr %71, ptr null)
  %76 = getelementptr inbounds float, ptr %1, i32 2
  %77 = load float, ptr %76, align 4
  %78 = getelementptr inbounds float, ptr %1, i32 1
  %79 = load float, ptr %78, align 4
  %80 = load float, ptr %1, align 4
  %81 = load float, ptr %32, align 4
  %82 = call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %55, i64 0)
  %83 = load ptr, ptr %82, align 8
  %84 = fpext float %81 to double
  call void @__catalyst__qis__RY(double %84, ptr %83, ptr null)
  %85 = fpext float %80 to double
  call void @__catalyst__qis__RZ(double %85, ptr %83, ptr null)
  %86 = fpext float %79 to double
  call void @__catalyst__qis__RY(double %86, ptr %83, ptr null)
  %87 = fpext float %77 to double
  call void @__catalyst__qis__RZ(double %87, ptr %83, ptr null)
  %88 = getelementptr inbounds float, ptr %1, i32 5
  %89 = load float, ptr %88, align 4
  %90 = getelementptr inbounds float, ptr %1, i32 4
  %91 = load float, ptr %90, align 4
  %92 = getelementptr inbounds float, ptr %1, i32 3
  %93 = load float, ptr %92, align 4
  %94 = getelementptr inbounds float, ptr %32, i32 1
  %95 = load float, ptr %94, align 4
  call void @_mlir_memref_to_llvm_free(ptr %27)
  %96 = call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %55, i64 1)
  %97 = load ptr, ptr %96, align 8
  %98 = fpext float %95 to double
  call void @__catalyst__qis__RY(double %98, ptr %97, ptr null)
  %99 = fpext float %93 to double
  call void @__catalyst__qis__RZ(double %99, ptr %97, ptr null)
  %100 = fpext float %91 to double
  call void @__catalyst__qis__RY(double %100, ptr %97, ptr null)
  %101 = fpext float %89 to double
  call void @__catalyst__qis__RZ(double %101, ptr %97, ptr null)
  call void @__catalyst__qis__CNOT(ptr %83, ptr %97, ptr null)
  call void @__catalyst__qis__CNOT(ptr %97, ptr %71, ptr null)
  call void @__catalyst__qis__CNOT(ptr %71, ptr %57, ptr null)
  call void @__catalyst__qis__CNOT(ptr %57, ptr %83, ptr null)
  %102 = fpext float %20 to double
  call void @__catalyst__qis__RZ(double %102, ptr %83, ptr null)
  %103 = fpext float %18 to double
  call void @__catalyst__qis__RY(double %103, ptr %83, ptr null)
  %104 = fpext float %16 to double
  call void @__catalyst__qis__RZ(double %104, ptr %83, ptr null)
  %105 = getelementptr inbounds float, ptr %1, i32 20
  %106 = load float, ptr %105, align 4
  %107 = getelementptr inbounds float, ptr %1, i32 19
  %108 = load float, ptr %107, align 4
  %109 = getelementptr inbounds float, ptr %1, i32 18
  %110 = load float, ptr %109, align 4
  %111 = fpext float %110 to double
  call void @__catalyst__qis__RZ(double %111, ptr %71, ptr null)
  %112 = fpext float %108 to double
  call void @__catalyst__qis__RY(double %112, ptr %71, ptr null)
  %113 = fpext float %106 to double
  call void @__catalyst__qis__RZ(double %113, ptr %71, ptr null)
  call void @__catalyst__qis__CNOT(ptr %83, ptr %71, ptr null)
  call void @__catalyst__qis__CNOT(ptr %71, ptr %83, ptr null)
  %114 = getelementptr inbounds float, ptr %1, i32 17
  %115 = load float, ptr %114, align 4
  %116 = getelementptr inbounds float, ptr %1, i32 16
  %117 = load float, ptr %116, align 4
  %118 = getelementptr inbounds float, ptr %1, i32 15
  %119 = load float, ptr %118, align 4
  %120 = fpext float %119 to double
  call void @__catalyst__qis__RZ(double %120, ptr %97, ptr null)
  %121 = fpext float %117 to double
  call void @__catalyst__qis__RY(double %121, ptr %97, ptr null)
  %122 = fpext float %115 to double
  call void @__catalyst__qis__RZ(double %122, ptr %97, ptr null)
  %123 = getelementptr inbounds float, ptr %1, i32 23
  %124 = load float, ptr %123, align 4
  %125 = getelementptr inbounds float, ptr %1, i32 22
  %126 = load float, ptr %125, align 4
  %127 = getelementptr inbounds float, ptr %1, i32 21
  %128 = load float, ptr %127, align 4
  %129 = fpext float %128 to double
  call void @__catalyst__qis__RZ(double %129, ptr %57, ptr null)
  %130 = fpext float %126 to double
  call void @__catalyst__qis__RY(double %130, ptr %57, ptr null)
  %131 = fpext float %124 to double
  call void @__catalyst__qis__RZ(double %131, ptr %57, ptr null)
  call void @__catalyst__qis__CNOT(ptr %97, ptr %57, ptr null)
  call void @__catalyst__qis__CNOT(ptr %57, ptr %97, ptr null)
  %132 = call i64 @__catalyst__qis__NamedObs(i64 3, ptr %83)
  %133 = call double @__catalyst__qis__Expval(i64 %132)
  %134 = call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %135 = ptrtoint ptr %134 to i64
  %136 = add i64 %135, 63
  %137 = urem i64 %136, 64
  %138 = sub i64 %136, %137
  %139 = inttoptr i64 %138 to ptr
  %140 = insertvalue { ptr, ptr, i64 } poison, ptr %134, 0
  %141 = insertvalue { ptr, ptr, i64 } %140, ptr %139, 1
  %142 = insertvalue { ptr, ptr, i64 } %141, i64 0, 2
  store double %133, ptr %139, align 8
  call void @__catalyst__rt__qubit_release_array(ptr %55)
  call void @__catalyst__rt__device_release()
  ret { ptr, ptr, i64 } %142
}

define void @setup() {
  call void @__catalyst__rt__initialize(ptr null)
  ret void
}

define void @teardown() {
  call void @__catalyst__rt__finalize()
  ret void
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #0

attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
