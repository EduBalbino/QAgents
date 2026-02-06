; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

@"{}" = internal constant [3 x i8] c"{}\00"
@LightningGPUSimulator = internal constant [22 x i8] c"LightningGPUSimulator\00"
@"/home/pichau/QAgents/.venv/lib/python3.12/site-packages/pennylane_lightning/liblightning_gpu_catalyst.so" = internal constant [105 x i8] c"/home/pichau/QAgents/.venv/lib/python3.12/site-packages/pennylane_lightning/liblightning_gpu_catalyst.so\00"
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
  %15 = getelementptr inbounds float, ptr %1, i32 74
  %16 = load float, ptr %15, align 4
  %17 = getelementptr inbounds float, ptr %1, i32 73
  %18 = load float, ptr %17, align 4
  %19 = getelementptr inbounds float, ptr %1, i32 72
  %20 = load float, ptr %19, align 4
  %21 = getelementptr inbounds float, ptr %1, i32 50
  %22 = load float, ptr %21, align 4
  %23 = getelementptr inbounds float, ptr %1, i32 49
  %24 = load float, ptr %23, align 4
  %25 = getelementptr inbounds float, ptr %1, i32 48
  %26 = load float, ptr %25, align 4
  %27 = getelementptr inbounds float, ptr %1, i32 44
  %28 = load float, ptr %27, align 4
  %29 = getelementptr inbounds float, ptr %1, i32 43
  %30 = load float, ptr %29, align 4
  %31 = getelementptr inbounds float, ptr %1, i32 42
  %32 = load float, ptr %31, align 4
  %33 = getelementptr inbounds float, ptr %1, i32 23
  %34 = load float, ptr %33, align 4
  %35 = getelementptr inbounds float, ptr %1, i32 22
  %36 = load float, ptr %35, align 4
  %37 = getelementptr inbounds float, ptr %1, i32 21
  %38 = load float, ptr %37, align 4
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
  %49 = load float, ptr @__constant_xf32, align 4
  %50 = getelementptr inbounds float, ptr %44, i64 %46
  store float %49, ptr %50, align 4
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
  %58 = load float, ptr %57, align 4
  %59 = getelementptr inbounds float, ptr %10, i64 %54
  %60 = load float, ptr %59, align 4
  %61 = fmul float %58, %60
  %62 = getelementptr inbounds float, ptr %44, i64 %54
  store float %61, ptr %62, align 4
  %63 = add i64 %54, 1
  br label %53

64:                                               ; preds = %53
  %65 = getelementptr inbounds float, ptr %44, i32 7
  %66 = load float, ptr %65, align 4
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
  %75 = load float, ptr %74, align 4
  %76 = getelementptr inbounds float, ptr %1, i32 19
  %77 = load float, ptr %76, align 4
  %78 = getelementptr inbounds float, ptr %1, i32 18
  %79 = load float, ptr %78, align 4
  %80 = getelementptr inbounds float, ptr %44, i32 6
  %81 = load float, ptr %80, align 4
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
  %89 = load float, ptr %88, align 4
  %90 = getelementptr inbounds float, ptr %1, i32 16
  %91 = load float, ptr %90, align 4
  %92 = getelementptr inbounds float, ptr %1, i32 15
  %93 = load float, ptr %92, align 4
  %94 = getelementptr inbounds float, ptr %44, i32 5
  %95 = load float, ptr %94, align 4
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
  %103 = load float, ptr %102, align 4
  %104 = getelementptr inbounds float, ptr %1, i32 13
  %105 = load float, ptr %104, align 4
  %106 = getelementptr inbounds float, ptr %1, i32 12
  %107 = load float, ptr %106, align 4
  %108 = getelementptr inbounds float, ptr %44, i32 4
  %109 = load float, ptr %108, align 4
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
  %117 = load float, ptr %116, align 4
  %118 = getelementptr inbounds float, ptr %1, i32 10
  %119 = load float, ptr %118, align 4
  %120 = getelementptr inbounds float, ptr %1, i32 9
  %121 = load float, ptr %120, align 4
  %122 = getelementptr inbounds float, ptr %44, i32 3
  %123 = load float, ptr %122, align 4
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
  %131 = load float, ptr %130, align 4
  %132 = getelementptr inbounds float, ptr %1, i32 7
  %133 = load float, ptr %132, align 4
  %134 = getelementptr inbounds float, ptr %1, i32 6
  %135 = load float, ptr %134, align 4
  %136 = getelementptr inbounds float, ptr %44, i32 2
  %137 = load float, ptr %136, align 4
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
  %145 = load float, ptr %144, align 4
  %146 = getelementptr inbounds float, ptr %1, i32 1
  %147 = load float, ptr %146, align 4
  %148 = load float, ptr %1, align 4
  %149 = load float, ptr %44, align 4
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
  %157 = load float, ptr %156, align 4
  %158 = getelementptr inbounds float, ptr %1, i32 4
  %159 = load float, ptr %158, align 4
  %160 = getelementptr inbounds float, ptr %1, i32 3
  %161 = load float, ptr %160, align 4
  %162 = getelementptr inbounds float, ptr %44, i32 1
  %163 = load float, ptr %162, align 4
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
  %174 = load float, ptr %173, align 4
  %175 = getelementptr inbounds float, ptr %1, i32 37
  %176 = load float, ptr %175, align 4
  %177 = getelementptr inbounds float, ptr %1, i32 36
  %178 = load float, ptr %177, align 4
  %179 = fpext float %178 to double
  call void @__catalyst__qis__RZ(double %179, ptr %111, ptr null)
  %180 = fpext float %176 to double
  call void @__catalyst__qis__RY(double %180, ptr %111, ptr null)
  %181 = fpext float %174 to double
  call void @__catalyst__qis__RZ(double %181, ptr %111, ptr null)
  %182 = getelementptr inbounds float, ptr %1, i32 26
  %183 = load float, ptr %182, align 4
  %184 = getelementptr inbounds float, ptr %1, i32 25
  %185 = load float, ptr %184, align 4
  %186 = getelementptr inbounds float, ptr %1, i32 24
  %187 = load float, ptr %186, align 4
  call void @__catalyst__qis__CNOT(ptr %69, ptr %151, ptr null)
  %188 = fpext float %187 to double
  call void @__catalyst__qis__RZ(double %188, ptr %151, ptr null)
  %189 = fpext float %185 to double
  call void @__catalyst__qis__RY(double %189, ptr %151, ptr null)
  %190 = fpext float %183 to double
  call void @__catalyst__qis__RZ(double %190, ptr %151, ptr null)
  %191 = getelementptr inbounds float, ptr %1, i32 32
  %192 = load float, ptr %191, align 4
  %193 = getelementptr inbounds float, ptr %1, i32 31
  %194 = load float, ptr %193, align 4
  %195 = getelementptr inbounds float, ptr %1, i32 30
  %196 = load float, ptr %195, align 4
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
  %204 = load float, ptr %203, align 4
  %205 = getelementptr inbounds float, ptr %1, i32 58
  %206 = load float, ptr %205, align 4
  %207 = getelementptr inbounds float, ptr %1, i32 57
  %208 = load float, ptr %207, align 4
  %209 = getelementptr inbounds float, ptr %1, i32 41
  %210 = load float, ptr %209, align 4
  %211 = getelementptr inbounds float, ptr %1, i32 40
  %212 = load float, ptr %211, align 4
  %213 = getelementptr inbounds float, ptr %1, i32 39
  %214 = load float, ptr %213, align 4
  %215 = fpext float %214 to double
  call void @__catalyst__qis__RZ(double %215, ptr %97, ptr null)
  %216 = fpext float %212 to double
  call void @__catalyst__qis__RY(double %216, ptr %97, ptr null)
  %217 = fpext float %210 to double
  call void @__catalyst__qis__RZ(double %217, ptr %97, ptr null)
  %218 = getelementptr inbounds float, ptr %1, i32 29
  %219 = load float, ptr %218, align 4
  %220 = getelementptr inbounds float, ptr %1, i32 28
  %221 = load float, ptr %220, align 4
  %222 = getelementptr inbounds float, ptr %1, i32 27
  %223 = load float, ptr %222, align 4
  %224 = fpext float %223 to double
  call void @__catalyst__qis__RZ(double %224, ptr %165, ptr null)
  %225 = fpext float %221 to double
  call void @__catalyst__qis__RY(double %225, ptr %165, ptr null)
  %226 = fpext float %219 to double
  call void @__catalyst__qis__RZ(double %226, ptr %165, ptr null)
  %227 = getelementptr inbounds float, ptr %1, i32 35
  %228 = load float, ptr %227, align 4
  %229 = getelementptr inbounds float, ptr %1, i32 34
  %230 = load float, ptr %229, align 4
  %231 = getelementptr inbounds float, ptr %1, i32 33
  %232 = load float, ptr %231, align 4
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
  %240 = load float, ptr %239, align 4
  %241 = getelementptr inbounds float, ptr %1, i32 55
  %242 = load float, ptr %241, align 4
  %243 = getelementptr inbounds float, ptr %1, i32 54
  %244 = load float, ptr %243, align 4
  %245 = fpext float %244 to double
  call void @__catalyst__qis__RZ(double %245, ptr %139, ptr null)
  %246 = fpext float %242 to double
  call void @__catalyst__qis__RY(double %246, ptr %139, ptr null)
  %247 = fpext float %240 to double
  call void @__catalyst__qis__RZ(double %247, ptr %139, ptr null)
  %248 = getelementptr inbounds float, ptr %1, i32 65
  %249 = load float, ptr %248, align 4
  %250 = getelementptr inbounds float, ptr %1, i32 64
  %251 = load float, ptr %250, align 4
  %252 = getelementptr inbounds float, ptr %1, i32 63
  %253 = load float, ptr %252, align 4
  %254 = getelementptr inbounds float, ptr %1, i32 47
  %255 = load float, ptr %254, align 4
  %256 = getelementptr inbounds float, ptr %1, i32 46
  %257 = load float, ptr %256, align 4
  %258 = getelementptr inbounds float, ptr %1, i32 45
  %259 = load float, ptr %258, align 4
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
  %270 = load float, ptr %269, align 4
  %271 = getelementptr inbounds float, ptr %1, i32 85
  %272 = load float, ptr %271, align 4
  %273 = getelementptr inbounds float, ptr %1, i32 84
  %274 = load float, ptr %273, align 4
  %275 = getelementptr inbounds float, ptr %1, i32 71
  %276 = load float, ptr %275, align 4
  %277 = getelementptr inbounds float, ptr %1, i32 70
  %278 = load float, ptr %277, align 4
  %279 = getelementptr inbounds float, ptr %1, i32 69
  %280 = load float, ptr %279, align 4
  call void @__catalyst__qis__CNOT(ptr %69, ptr %165, ptr null)
  %281 = fpext float %280 to double
  call void @__catalyst__qis__RZ(double %281, ptr %69, ptr null)
  %282 = fpext float %278 to double
  call void @__catalyst__qis__RY(double %282, ptr %69, ptr null)
  %283 = fpext float %276 to double
  call void @__catalyst__qis__RZ(double %283, ptr %69, ptr null)
  %284 = getelementptr inbounds float, ptr %1, i32 53
  %285 = load float, ptr %284, align 4
  %286 = getelementptr inbounds float, ptr %1, i32 52
  %287 = load float, ptr %286, align 4
  %288 = getelementptr inbounds float, ptr %1, i32 51
  %289 = load float, ptr %288, align 4
  %290 = fpext float %289 to double
  call void @__catalyst__qis__RZ(double %290, ptr %165, ptr null)
  %291 = fpext float %287 to double
  call void @__catalyst__qis__RY(double %291, ptr %165, ptr null)
  %292 = fpext float %285 to double
  call void @__catalyst__qis__RZ(double %292, ptr %165, ptr null)
  %293 = getelementptr inbounds float, ptr %1, i32 62
  %294 = load float, ptr %293, align 4
  %295 = getelementptr inbounds float, ptr %1, i32 61
  %296 = load float, ptr %295, align 4
  %297 = getelementptr inbounds float, ptr %1, i32 60
  %298 = load float, ptr %297, align 4
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
  %306 = load float, ptr %305, align 4
  %307 = getelementptr inbounds float, ptr %1, i32 76
  %308 = load float, ptr %307, align 4
  %309 = getelementptr inbounds float, ptr %1, i32 75
  %310 = load float, ptr %309, align 4
  %311 = getelementptr inbounds float, ptr %1, i32 68
  %312 = load float, ptr %311, align 4
  %313 = getelementptr inbounds float, ptr %1, i32 67
  %314 = load float, ptr %313, align 4
  %315 = getelementptr inbounds float, ptr %1, i32 66
  %316 = load float, ptr %315, align 4
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
  %324 = load float, ptr %323, align 4
  %325 = getelementptr inbounds float, ptr %1, i32 88
  %326 = load float, ptr %325, align 4
  %327 = getelementptr inbounds float, ptr %1, i32 87
  %328 = load float, ptr %327, align 4
  %329 = fpext float %328 to double
  call void @__catalyst__qis__RZ(double %329, ptr %97, ptr null)
  %330 = fpext float %326 to double
  call void @__catalyst__qis__RY(double %330, ptr %97, ptr null)
  %331 = fpext float %324 to double
  call void @__catalyst__qis__RZ(double %331, ptr %97, ptr null)
  call void @__catalyst__qis__CNOT(ptr %165, ptr %97, ptr null)
  call void @__catalyst__qis__CNOT(ptr %97, ptr %165, ptr null)
  %332 = getelementptr inbounds float, ptr %1, i32 80
  %333 = load float, ptr %332, align 4
  %334 = getelementptr inbounds float, ptr %1, i32 79
  %335 = load float, ptr %334, align 4
  %336 = getelementptr inbounds float, ptr %1, i32 78
  %337 = load float, ptr %336, align 4
  call void @__catalyst__qis__CNOT(ptr %69, ptr %139, ptr null)
  %338 = fpext float %337 to double
  call void @__catalyst__qis__RZ(double %338, ptr %139, ptr null)
  %339 = fpext float %335 to double
  call void @__catalyst__qis__RY(double %339, ptr %139, ptr null)
  %340 = fpext float %333 to double
  call void @__catalyst__qis__RZ(double %340, ptr %139, ptr null)
  %341 = getelementptr inbounds float, ptr %1, i32 92
  %342 = load float, ptr %341, align 4
  %343 = getelementptr inbounds float, ptr %1, i32 91
  %344 = load float, ptr %343, align 4
  %345 = getelementptr inbounds float, ptr %1, i32 90
  %346 = load float, ptr %345, align 4
  %347 = fpext float %346 to double
  call void @__catalyst__qis__RZ(double %347, ptr %83, ptr null)
  %348 = fpext float %344 to double
  call void @__catalyst__qis__RY(double %348, ptr %83, ptr null)
  %349 = fpext float %342 to double
  call void @__catalyst__qis__RZ(double %349, ptr %83, ptr null)
  call void @__catalyst__qis__CNOT(ptr %139, ptr %83, ptr null)
  call void @__catalyst__qis__CNOT(ptr %83, ptr %139, ptr null)
  %350 = getelementptr inbounds float, ptr %1, i32 83
  %351 = load float, ptr %350, align 4
  %352 = getelementptr inbounds float, ptr %1, i32 82
  %353 = load float, ptr %352, align 4
  %354 = getelementptr inbounds float, ptr %1, i32 81
  %355 = load float, ptr %354, align 4
  %356 = fpext float %355 to double
  call void @__catalyst__qis__RZ(double %356, ptr %125, ptr null)
  %357 = fpext float %353 to double
  call void @__catalyst__qis__RY(double %357, ptr %125, ptr null)
  %358 = fpext float %351 to double
  call void @__catalyst__qis__RZ(double %358, ptr %125, ptr null)
  %359 = getelementptr inbounds float, ptr %1, i32 95
  %360 = load float, ptr %359, align 4
  %361 = getelementptr inbounds float, ptr %1, i32 94
  %362 = load float, ptr %361, align 4
  %363 = getelementptr inbounds float, ptr %1, i32 93
  %364 = load float, ptr %363, align 4
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
  store double %369, ptr %375, align 8
  call void @__catalyst__rt__qubit_release_array(ptr %67)
  call void @__catalyst__rt__device_release()
  ret { ptr, ptr, i64 } %378
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
