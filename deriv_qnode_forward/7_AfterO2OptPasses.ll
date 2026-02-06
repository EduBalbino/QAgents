; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@"{}" = internal constant [3 x i8] c"{}\00"
@LightningGPUSimulator = internal constant [22 x i8] c"LightningGPUSimulator\00"
@"/home/pichau/QAgents/.venv/lib/python3.12/site-packages/pennylane_lightning/liblightning_gpu_catalyst.so" = internal constant [105 x i8] c"/home/pichau/QAgents/.venv/lib/python3.12/site-packages/pennylane_lightning/liblightning_gpu_catalyst.so\00"
@enzyme_dupnoneed = linkonce constant i8 0
@enzyme_const = linkonce constant i8 0
@__enzyme_function_like_free = local_unnamed_addr global [2 x ptr] [ptr @_mlir_memref_to_llvm_free, ptr @freename]
@freename = linkonce constant [5 x i8] c"free\00"
@dealloc_indices = linkonce constant [3 x i8] c"-1\00"
@__enzyme_allocation_like = local_unnamed_addr global [4 x ptr] [ptr @_mlir_memref_to_llvm_alloc, ptr null, ptr @dealloc_indices, ptr @_mlir_memref_to_llvm_free]
@__enzyme_register_gradient_qnode_forward_0.quantum = local_unnamed_addr global [3 x ptr] [ptr @qnode_forward_0.quantum, ptr @qnode_forward_0.quantum.augfwd, ptr @qnode_forward_0.quantum.customqgrad]

declare void @__catalyst__rt__finalize() local_unnamed_addr

declare void @__catalyst__rt__initialize(ptr) local_unnamed_addr

declare void @__catalyst__rt__device_release() local_unnamed_addr

declare void @__catalyst__rt__qubit_release_array(ptr) local_unnamed_addr

declare double @__catalyst__qis__Expval(i64) local_unnamed_addr

declare i64 @__catalyst__qis__NamedObs(i64, ptr) local_unnamed_addr

declare void @__catalyst__qis__CNOT(ptr, ptr, ptr) local_unnamed_addr

declare void @__catalyst__qis__RZ(double, ptr, ptr) local_unnamed_addr

declare void @__catalyst__qis__RY(double, ptr, ptr) local_unnamed_addr

declare ptr @__catalyst__rt__array_get_element_ptr_1d(ptr, i64) local_unnamed_addr

declare ptr @__catalyst__rt__qubit_allocate_array(i64) local_unnamed_addr

declare void @__catalyst__rt__device_init(ptr, ptr, ptr, i64, i1) local_unnamed_addr

declare void @__catalyst__qis__Gradient(i64, ...) local_unnamed_addr

declare void @__catalyst__rt__toggle_recorder(i1) local_unnamed_addr

declare void @__enzyme_autodiff0(...) local_unnamed_addr

declare void @_mlir_memref_to_llvm_free(ptr)

declare ptr @_mlir_memref_to_llvm_alloc(i64)

define { ptr, ptr, i64, [3 x i64], [3 x i64] } @jit_deriv_qnode_forward(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, ptr %9, ptr %10, i64 %11, i64 %12, i64 %13) local_unnamed_addr {
  %15 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %16 = ptrtoint ptr %15 to i64
  %17 = add i64 %16, 63
  %18 = and i64 %17, -64
  %19 = inttoptr i64 %18 to ptr
  store double 1.000000e+00, ptr %19, align 64, !tbaa !1
  %20 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 384)
  %21 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 8)
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 1 dereferenceable(384) %20, i8 0, i64 384, i1 false)
  tail call void (...) @__enzyme_autodiff0(ptr nonnull @qnode_forward_0.preprocess, ptr nonnull @enzyme_const, ptr %0, ptr %1, ptr nonnull %20, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, ptr nonnull @enzyme_const, ptr %9, ptr nonnull @enzyme_const, ptr %10, i64 %11, i64 %12, i64 %13, ptr nonnull @enzyme_const, i64 104, ptr nonnull @enzyme_const, ptr %21, ptr nonnull @enzyme_dupnoneed, ptr %21, ptr nonnull %19, i64 0)
  tail call void @_mlir_memref_to_llvm_free(ptr %21)
  tail call void @_mlir_memref_to_llvm_free(ptr %15)
  %22 = icmp eq ptr %20, inttoptr (i64 3735928559 to ptr)
  br i1 %22, label %23, label %25

23:                                               ; preds = %14
  %24 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 384)
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(384) %24, ptr noundef nonnull align 1 dereferenceable(384) inttoptr (i64 3735928559 to ptr), i64 384, i1 false)
  br label %25

25:                                               ; preds = %23, %14
  %.pn15 = phi ptr [ %24, %23 ], [ %20, %14 ]
  %.pn14 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } poison, ptr %.pn15, 0
  %.pn12 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %.pn14, ptr %.pn15, 1
  %.pn10 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %.pn12, i64 0, 2
  %.pn8 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %.pn10, i64 4, 3, 0
  %.pn6 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %.pn8, i64 8, 3, 1
  %.pn4 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %.pn6, i64 3, 3, 2
  %.pn2 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %.pn4, i64 24, 4, 0
  %.pn = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %.pn2, i64 3, 4, 1
  %26 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %.pn, i64 1, 4, 2
  ret { ptr, ptr, i64, [3 x i64], [3 x i64] } %26
}

define void @_catalyst_pyface_jit_deriv_qnode_forward(ptr writeonly captures(none) initializes((0, 72)) %0, ptr readonly captures(none) %1) local_unnamed_addr {
  %.unpack = load ptr, ptr %1, align 8
  %.elt1 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %.unpack2 = load ptr, ptr %.elt1, align 8
  %.unpack.i = load ptr, ptr %.unpack, align 8
  %.elt1.i = getelementptr inbounds nuw i8, ptr %.unpack, i64 8
  %.unpack2.i = load ptr, ptr %.elt1.i, align 8
  %.elt3.i = getelementptr inbounds nuw i8, ptr %.unpack, i64 16
  %.unpack4.i = load i64, ptr %.elt3.i, align 8
  %.elt5.i = getelementptr inbounds nuw i8, ptr %.unpack, i64 24
  %.unpack6.unpack.i = load i64, ptr %.elt5.i, align 8
  %.unpack6.elt9.i = getelementptr inbounds nuw i8, ptr %.unpack, i64 32
  %.unpack6.unpack10.i = load i64, ptr %.unpack6.elt9.i, align 8
  %.unpack6.elt11.i = getelementptr inbounds nuw i8, ptr %.unpack, i64 40
  %.unpack6.unpack12.i = load i64, ptr %.unpack6.elt11.i, align 8
  %.elt7.i = getelementptr inbounds nuw i8, ptr %.unpack, i64 48
  %.unpack8.unpack.i = load i64, ptr %.elt7.i, align 8
  %.unpack8.elt14.i = getelementptr inbounds nuw i8, ptr %.unpack, i64 56
  %.unpack8.unpack15.i = load i64, ptr %.unpack8.elt14.i, align 8
  %.unpack8.elt16.i = getelementptr inbounds nuw i8, ptr %.unpack, i64 64
  %.unpack8.unpack17.i = load i64, ptr %.unpack8.elt16.i, align 8
  %.unpack19.i = load ptr, ptr %.unpack2, align 8
  %.elt20.i = getelementptr inbounds nuw i8, ptr %.unpack2, i64 8
  %.unpack21.i = load ptr, ptr %.elt20.i, align 8
  %.elt22.i = getelementptr inbounds nuw i8, ptr %.unpack2, i64 16
  %.unpack23.i = load i64, ptr %.elt22.i, align 8
  %.elt24.i = getelementptr inbounds nuw i8, ptr %.unpack2, i64 24
  %.unpack25.unpack.i = load i64, ptr %.elt24.i, align 8
  %.elt26.i = getelementptr inbounds nuw i8, ptr %.unpack2, i64 32
  %.unpack27.unpack.i = load i64, ptr %.elt26.i, align 8
  %3 = tail call { ptr, ptr, i64, [3 x i64], [3 x i64] } @jit_deriv_qnode_forward(ptr %.unpack.i, ptr %.unpack2.i, i64 %.unpack4.i, i64 %.unpack6.unpack.i, i64 %.unpack6.unpack10.i, i64 %.unpack6.unpack12.i, i64 %.unpack8.unpack.i, i64 %.unpack8.unpack15.i, i64 %.unpack8.unpack17.i, ptr %.unpack19.i, ptr %.unpack21.i, i64 %.unpack23.i, i64 %.unpack25.unpack.i, i64 %.unpack27.unpack.i)
  %.elt.i = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %3, 0
  store ptr %.elt.i, ptr %0, align 8
  %.repack30.i = getelementptr inbounds nuw i8, ptr %0, i64 8
  %.elt31.i = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %3, 1
  store ptr %.elt31.i, ptr %.repack30.i, align 8
  %.repack32.i = getelementptr inbounds nuw i8, ptr %0, i64 16
  %.elt33.i = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %3, 2
  store i64 %.elt33.i, ptr %.repack32.i, align 8
  %.repack34.i = getelementptr inbounds nuw i8, ptr %0, i64 24
  %.elt35.i = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %3, 3
  %.elt35.elt.i = extractvalue [3 x i64] %.elt35.i, 0
  store i64 %.elt35.elt.i, ptr %.repack34.i, align 8
  %.repack34.repack38.i = getelementptr inbounds nuw i8, ptr %0, i64 32
  %.elt35.elt39.i = extractvalue [3 x i64] %.elt35.i, 1
  store i64 %.elt35.elt39.i, ptr %.repack34.repack38.i, align 8
  %.repack34.repack40.i = getelementptr inbounds nuw i8, ptr %0, i64 40
  %.elt35.elt41.i = extractvalue [3 x i64] %.elt35.i, 2
  store i64 %.elt35.elt41.i, ptr %.repack34.repack40.i, align 8
  %.repack36.i = getelementptr inbounds nuw i8, ptr %0, i64 48
  %.elt37.i = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %3, 4
  %.elt37.elt.i = extractvalue [3 x i64] %.elt37.i, 0
  store i64 %.elt37.elt.i, ptr %.repack36.i, align 8
  %.repack36.repack42.i = getelementptr inbounds nuw i8, ptr %0, i64 56
  %.elt37.elt43.i = extractvalue [3 x i64] %.elt37.i, 1
  store i64 %.elt37.elt43.i, ptr %.repack36.repack42.i, align 8
  %.repack36.repack44.i = getelementptr inbounds nuw i8, ptr %0, i64 64
  %.elt37.elt45.i = extractvalue [3 x i64] %.elt37.i, 2
  store i64 %.elt37.elt45.i, ptr %.repack36.repack44.i, align 8
  ret void
}

define void @_catalyst_ciface_jit_deriv_qnode_forward(ptr writeonly captures(none) initializes((0, 72)) %0, ptr readonly captures(none) %1, ptr readonly captures(none) %2) local_unnamed_addr {
  %.unpack = load ptr, ptr %1, align 8
  %.elt1 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %.unpack2 = load ptr, ptr %.elt1, align 8
  %.elt3 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %.unpack4 = load i64, ptr %.elt3, align 8
  %.elt5 = getelementptr inbounds nuw i8, ptr %1, i64 24
  %.unpack6.unpack = load i64, ptr %.elt5, align 8
  %.unpack6.elt9 = getelementptr inbounds nuw i8, ptr %1, i64 32
  %.unpack6.unpack10 = load i64, ptr %.unpack6.elt9, align 8
  %.unpack6.elt11 = getelementptr inbounds nuw i8, ptr %1, i64 40
  %.unpack6.unpack12 = load i64, ptr %.unpack6.elt11, align 8
  %.elt7 = getelementptr inbounds nuw i8, ptr %1, i64 48
  %.unpack8.unpack = load i64, ptr %.elt7, align 8
  %.unpack8.elt14 = getelementptr inbounds nuw i8, ptr %1, i64 56
  %.unpack8.unpack15 = load i64, ptr %.unpack8.elt14, align 8
  %.unpack8.elt16 = getelementptr inbounds nuw i8, ptr %1, i64 64
  %.unpack8.unpack17 = load i64, ptr %.unpack8.elt16, align 8
  %.unpack19 = load ptr, ptr %2, align 8
  %.elt20 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %.unpack21 = load ptr, ptr %.elt20, align 8
  %.elt22 = getelementptr inbounds nuw i8, ptr %2, i64 16
  %.unpack23 = load i64, ptr %.elt22, align 8
  %.elt24 = getelementptr inbounds nuw i8, ptr %2, i64 24
  %.unpack25.unpack = load i64, ptr %.elt24, align 8
  %.elt26 = getelementptr inbounds nuw i8, ptr %2, i64 32
  %.unpack27.unpack = load i64, ptr %.elt26, align 8
  %4 = tail call { ptr, ptr, i64, [3 x i64], [3 x i64] } @jit_deriv_qnode_forward(ptr %.unpack, ptr %.unpack2, i64 %.unpack4, i64 %.unpack6.unpack, i64 %.unpack6.unpack10, i64 %.unpack6.unpack12, i64 %.unpack8.unpack, i64 %.unpack8.unpack15, i64 %.unpack8.unpack17, ptr %.unpack19, ptr %.unpack21, i64 %.unpack23, i64 %.unpack25.unpack, i64 %.unpack27.unpack)
  %.elt = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %4, 0
  store ptr %.elt, ptr %0, align 8
  %.repack30 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %.elt31 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %4, 1
  store ptr %.elt31, ptr %.repack30, align 8
  %.repack32 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %.elt33 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %4, 2
  store i64 %.elt33, ptr %.repack32, align 8
  %.repack34 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %.elt35 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %4, 3
  %.elt35.elt = extractvalue [3 x i64] %.elt35, 0
  store i64 %.elt35.elt, ptr %.repack34, align 8
  %.repack34.repack38 = getelementptr inbounds nuw i8, ptr %0, i64 32
  %.elt35.elt39 = extractvalue [3 x i64] %.elt35, 1
  store i64 %.elt35.elt39, ptr %.repack34.repack38, align 8
  %.repack34.repack40 = getelementptr inbounds nuw i8, ptr %0, i64 40
  %.elt35.elt41 = extractvalue [3 x i64] %.elt35, 2
  store i64 %.elt35.elt41, ptr %.repack34.repack40, align 8
  %.repack36 = getelementptr inbounds nuw i8, ptr %0, i64 48
  %.elt37 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %4, 4
  %.elt37.elt = extractvalue [3 x i64] %.elt37, 0
  store i64 %.elt37.elt, ptr %.repack36, align 8
  %.repack36.repack42 = getelementptr inbounds nuw i8, ptr %0, i64 56
  %.elt37.elt43 = extractvalue [3 x i64] %.elt37, 1
  store i64 %.elt37.elt43, ptr %.repack36.repack42, align 8
  %.repack36.repack44 = getelementptr inbounds nuw i8, ptr %0, i64 64
  %.elt37.elt45 = extractvalue [3 x i64] %.elt37, 2
  store i64 %.elt37.elt45, ptr %.repack36.repack44, align 8
  ret void
}

define { ptr, ptr, i64, [1 x i64], [1 x i64] } @qnode_forward_0.adjoint(ptr readnone captures(none) %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, ptr readnone captures(none) %9, ptr readonly captures(none) %10, i64 %11, i64 %12, i64 %13, i64 %14) local_unnamed_addr {
  %.idx = shl i64 %14, 3
  %16 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 %.idx)
  %17 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %16, 0
  %18 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %17, ptr %16, 1
  %19 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %18, i64 0, 2
  %20 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %19, i64 %14, 3, 0
  %21 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, i64 1, 4, 0
  tail call void @__catalyst__rt__toggle_recorder(i1 true)
  %22 = tail call { ptr, double } @qnode_forward_0.nodealloc(ptr poison, ptr %1, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, ptr poison, ptr %10, i64 poison, i64 poison, i64 poison)
  %23 = extractvalue { ptr, double } %22, 0
  tail call void @__catalyst__rt__toggle_recorder(i1 false)
  %24 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, align 8
  store ptr %16, ptr %24, align 8
  %.fca.1.gep = getelementptr inbounds nuw i8, ptr %24, i64 8
  store ptr %16, ptr %.fca.1.gep, align 8
  %.fca.2.gep = getelementptr inbounds nuw i8, ptr %24, i64 16
  store i64 0, ptr %.fca.2.gep, align 8
  %.fca.3.0.gep = getelementptr inbounds nuw i8, ptr %24, i64 24
  store i64 %14, ptr %.fca.3.0.gep, align 8
  %.fca.4.0.gep = getelementptr inbounds nuw i8, ptr %24, i64 32
  store i64 1, ptr %.fca.4.0.gep, align 8
  call void (i64, ...) @__catalyst__qis__Gradient(i64 1, ptr nonnull %24)
  call void @__catalyst__rt__qubit_release_array(ptr %23)
  call void @__catalyst__rt__device_release()
  ret { ptr, ptr, i64, [1 x i64], [1 x i64] } %21
}

define { ptr, double } @qnode_forward_0.nodealloc(ptr readnone captures(none) %0, ptr readonly %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, ptr readnone captures(none) %9, ptr readonly captures(none) %10, i64 %11, i64 %12, i64 %13) local_unnamed_addr {
.preheader.preheader:
  %14 = getelementptr inbounds nuw i8, ptr %1, i64 296
  %15 = load float, ptr %14, align 4, !tbaa !4
  %16 = getelementptr inbounds nuw i8, ptr %1, i64 292
  %17 = load float, ptr %16, align 4, !tbaa !4
  %18 = getelementptr inbounds nuw i8, ptr %1, i64 288
  %19 = load float, ptr %18, align 4, !tbaa !4
  %20 = getelementptr inbounds nuw i8, ptr %1, i64 200
  %21 = load float, ptr %20, align 4, !tbaa !4
  %22 = getelementptr inbounds nuw i8, ptr %1, i64 196
  %23 = load float, ptr %22, align 4, !tbaa !4
  %24 = getelementptr inbounds nuw i8, ptr %1, i64 192
  %25 = load float, ptr %24, align 4, !tbaa !4
  %26 = getelementptr inbounds nuw i8, ptr %1, i64 176
  %27 = load float, ptr %26, align 4, !tbaa !4
  %28 = getelementptr inbounds nuw i8, ptr %1, i64 172
  %29 = load float, ptr %28, align 4, !tbaa !4
  %30 = getelementptr inbounds nuw i8, ptr %1, i64 168
  %31 = load float, ptr %30, align 4, !tbaa !4
  %32 = getelementptr inbounds nuw i8, ptr %1, i64 92
  %33 = load float, ptr %32, align 4, !tbaa !4
  %34 = getelementptr inbounds nuw i8, ptr %1, i64 88
  %35 = load float, ptr %34, align 4, !tbaa !4
  %36 = getelementptr inbounds nuw i8, ptr %1, i64 84
  %37 = load float, ptr %36, align 4, !tbaa !4
  %38 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 96)
  %39 = ptrtoint ptr %38 to i64
  %40 = add i64 %39, 63
  %41 = and i64 %40, -64
  %42 = inttoptr i64 %41 to ptr
  store float 0x400921FB60000000, ptr %42, align 64, !tbaa !4
  %43 = getelementptr inbounds nuw i8, ptr %42, i64 4
  store float 0x400921FB60000000, ptr %43, align 4, !tbaa !4
  %44 = getelementptr inbounds nuw i8, ptr %42, i64 8
  store float 0x400921FB60000000, ptr %44, align 8, !tbaa !4
  %45 = getelementptr inbounds nuw i8, ptr %42, i64 12
  store float 0x400921FB60000000, ptr %45, align 4, !tbaa !4
  %46 = getelementptr inbounds nuw i8, ptr %42, i64 16
  store float 0x400921FB60000000, ptr %46, align 16, !tbaa !4
  %47 = getelementptr inbounds nuw i8, ptr %42, i64 20
  store float 0x400921FB60000000, ptr %47, align 4, !tbaa !4
  %48 = getelementptr inbounds nuw i8, ptr %42, i64 24
  store float 0x400921FB60000000, ptr %48, align 8, !tbaa !4
  %49 = getelementptr inbounds nuw i8, ptr %42, i64 28
  store float 0x400921FB60000000, ptr %49, align 4, !tbaa !4
  %50 = load float, ptr %10, align 4, !tbaa !4
  %51 = fmul float %50, 0x400921FB60000000
  store float %51, ptr %42, align 64, !tbaa !4
  %52 = getelementptr inbounds nuw i8, ptr %10, i64 4
  %53 = load float, ptr %52, align 4, !tbaa !4
  %54 = fmul float %53, 0x400921FB60000000
  store float %54, ptr %43, align 4, !tbaa !4
  %55 = getelementptr inbounds nuw i8, ptr %10, i64 8
  %56 = load float, ptr %55, align 4, !tbaa !4
  %57 = fmul float %56, 0x400921FB60000000
  store float %57, ptr %44, align 8, !tbaa !4
  %58 = getelementptr inbounds nuw i8, ptr %10, i64 12
  %59 = load float, ptr %58, align 4, !tbaa !4
  %60 = fmul float %59, 0x400921FB60000000
  store float %60, ptr %45, align 4, !tbaa !4
  %61 = getelementptr inbounds nuw i8, ptr %10, i64 16
  %62 = load float, ptr %61, align 4, !tbaa !4
  %63 = fmul float %62, 0x400921FB60000000
  store float %63, ptr %46, align 16, !tbaa !4
  %64 = getelementptr inbounds nuw i8, ptr %10, i64 20
  %65 = load float, ptr %64, align 4, !tbaa !4
  %66 = fmul float %65, 0x400921FB60000000
  store float %66, ptr %47, align 4, !tbaa !4
  %67 = getelementptr inbounds nuw i8, ptr %10, i64 24
  %68 = load float, ptr %67, align 4, !tbaa !4
  %69 = fmul float %68, 0x400921FB60000000
  store float %69, ptr %48, align 8, !tbaa !4
  %70 = getelementptr inbounds nuw i8, ptr %10, i64 28
  %71 = load float, ptr %70, align 4, !tbaa !4
  %72 = fmul float %71, 0x400921FB60000000
  store float %72, ptr %49, align 4, !tbaa !4
  tail call void @__catalyst__rt__device_init(ptr nonnull @"/home/pichau/QAgents/.venv/lib/python3.12/site-packages/pennylane_lightning/liblightning_gpu_catalyst.so", ptr nonnull @LightningGPUSimulator, ptr nonnull @"{}", i64 0, i1 false)
  %73 = tail call ptr @__catalyst__rt__qubit_allocate_array(i64 8)
  %74 = tail call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %73, i64 7)
  %75 = load ptr, ptr %74, align 8
  %76 = fpext float %72 to double
  tail call void @__catalyst__qis__RY(double %76, ptr %75, ptr null)
  %77 = fpext float %37 to double
  tail call void @__catalyst__qis__RZ(double %77, ptr %75, ptr null)
  %78 = fpext float %35 to double
  tail call void @__catalyst__qis__RY(double %78, ptr %75, ptr null)
  %79 = fpext float %33 to double
  tail call void @__catalyst__qis__RZ(double %79, ptr %75, ptr null)
  %80 = getelementptr inbounds nuw i8, ptr %1, i64 80
  %81 = load float, ptr %80, align 4, !tbaa !4
  %82 = getelementptr inbounds nuw i8, ptr %1, i64 76
  %83 = load float, ptr %82, align 4, !tbaa !4
  %84 = getelementptr inbounds nuw i8, ptr %1, i64 72
  %85 = load float, ptr %84, align 4, !tbaa !4
  %86 = load float, ptr %48, align 8, !tbaa !4
  %87 = tail call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %73, i64 6)
  %88 = load ptr, ptr %87, align 8
  %89 = fpext float %86 to double
  tail call void @__catalyst__qis__RY(double %89, ptr %88, ptr null)
  %90 = fpext float %85 to double
  tail call void @__catalyst__qis__RZ(double %90, ptr %88, ptr null)
  %91 = fpext float %83 to double
  tail call void @__catalyst__qis__RY(double %91, ptr %88, ptr null)
  %92 = fpext float %81 to double
  tail call void @__catalyst__qis__RZ(double %92, ptr %88, ptr null)
  %93 = getelementptr inbounds nuw i8, ptr %1, i64 68
  %94 = load float, ptr %93, align 4, !tbaa !4
  %95 = getelementptr inbounds nuw i8, ptr %1, i64 64
  %96 = load float, ptr %95, align 4, !tbaa !4
  %97 = getelementptr inbounds nuw i8, ptr %1, i64 60
  %98 = load float, ptr %97, align 4, !tbaa !4
  %99 = load float, ptr %47, align 4, !tbaa !4
  %100 = tail call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %73, i64 5)
  %101 = load ptr, ptr %100, align 8
  %102 = fpext float %99 to double
  tail call void @__catalyst__qis__RY(double %102, ptr %101, ptr null)
  %103 = fpext float %98 to double
  tail call void @__catalyst__qis__RZ(double %103, ptr %101, ptr null)
  %104 = fpext float %96 to double
  tail call void @__catalyst__qis__RY(double %104, ptr %101, ptr null)
  %105 = fpext float %94 to double
  tail call void @__catalyst__qis__RZ(double %105, ptr %101, ptr null)
  %106 = getelementptr inbounds nuw i8, ptr %1, i64 56
  %107 = load float, ptr %106, align 4, !tbaa !4
  %108 = getelementptr inbounds nuw i8, ptr %1, i64 52
  %109 = load float, ptr %108, align 4, !tbaa !4
  %110 = getelementptr inbounds nuw i8, ptr %1, i64 48
  %111 = load float, ptr %110, align 4, !tbaa !4
  %112 = load float, ptr %46, align 16, !tbaa !4
  %113 = tail call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %73, i64 4)
  %114 = load ptr, ptr %113, align 8
  %115 = fpext float %112 to double
  tail call void @__catalyst__qis__RY(double %115, ptr %114, ptr null)
  %116 = fpext float %111 to double
  tail call void @__catalyst__qis__RZ(double %116, ptr %114, ptr null)
  %117 = fpext float %109 to double
  tail call void @__catalyst__qis__RY(double %117, ptr %114, ptr null)
  %118 = fpext float %107 to double
  tail call void @__catalyst__qis__RZ(double %118, ptr %114, ptr null)
  %119 = getelementptr inbounds nuw i8, ptr %1, i64 44
  %120 = load float, ptr %119, align 4, !tbaa !4
  %121 = getelementptr inbounds nuw i8, ptr %1, i64 40
  %122 = load float, ptr %121, align 4, !tbaa !4
  %123 = getelementptr inbounds nuw i8, ptr %1, i64 36
  %124 = load float, ptr %123, align 4, !tbaa !4
  %125 = load float, ptr %45, align 4, !tbaa !4
  %126 = tail call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %73, i64 3)
  %127 = load ptr, ptr %126, align 8
  %128 = fpext float %125 to double
  tail call void @__catalyst__qis__RY(double %128, ptr %127, ptr null)
  %129 = fpext float %124 to double
  tail call void @__catalyst__qis__RZ(double %129, ptr %127, ptr null)
  %130 = fpext float %122 to double
  tail call void @__catalyst__qis__RY(double %130, ptr %127, ptr null)
  %131 = fpext float %120 to double
  tail call void @__catalyst__qis__RZ(double %131, ptr %127, ptr null)
  %132 = getelementptr inbounds nuw i8, ptr %1, i64 32
  %133 = load float, ptr %132, align 4, !tbaa !4
  %134 = getelementptr inbounds nuw i8, ptr %1, i64 28
  %135 = load float, ptr %134, align 4, !tbaa !4
  %136 = getelementptr inbounds nuw i8, ptr %1, i64 24
  %137 = load float, ptr %136, align 4, !tbaa !4
  %138 = load float, ptr %44, align 8, !tbaa !4
  %139 = tail call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %73, i64 2)
  %140 = load ptr, ptr %139, align 8
  %141 = fpext float %138 to double
  tail call void @__catalyst__qis__RY(double %141, ptr %140, ptr null)
  %142 = fpext float %137 to double
  tail call void @__catalyst__qis__RZ(double %142, ptr %140, ptr null)
  %143 = fpext float %135 to double
  tail call void @__catalyst__qis__RY(double %143, ptr %140, ptr null)
  %144 = fpext float %133 to double
  tail call void @__catalyst__qis__RZ(double %144, ptr %140, ptr null)
  %145 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %146 = load float, ptr %145, align 4, !tbaa !4
  %147 = getelementptr inbounds nuw i8, ptr %1, i64 4
  %148 = load float, ptr %147, align 4, !tbaa !4
  %149 = load float, ptr %1, align 4, !tbaa !4
  %150 = load float, ptr %42, align 64, !tbaa !4
  %151 = tail call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %73, i64 0)
  %152 = load ptr, ptr %151, align 8
  %153 = fpext float %150 to double
  tail call void @__catalyst__qis__RY(double %153, ptr %152, ptr null)
  %154 = fpext float %149 to double
  tail call void @__catalyst__qis__RZ(double %154, ptr %152, ptr null)
  %155 = fpext float %148 to double
  tail call void @__catalyst__qis__RY(double %155, ptr %152, ptr null)
  %156 = fpext float %146 to double
  tail call void @__catalyst__qis__RZ(double %156, ptr %152, ptr null)
  %157 = getelementptr inbounds nuw i8, ptr %1, i64 20
  %158 = load float, ptr %157, align 4, !tbaa !4
  %159 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %160 = load float, ptr %159, align 4, !tbaa !4
  %161 = getelementptr inbounds nuw i8, ptr %1, i64 12
  %162 = load float, ptr %161, align 4, !tbaa !4
  %163 = load float, ptr %43, align 4, !tbaa !4
  tail call void @_mlir_memref_to_llvm_free(ptr %38)
  %164 = tail call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %73, i64 1)
  %165 = load ptr, ptr %164, align 8
  %166 = fpext float %163 to double
  tail call void @__catalyst__qis__RY(double %166, ptr %165, ptr null)
  %167 = fpext float %162 to double
  tail call void @__catalyst__qis__RZ(double %167, ptr %165, ptr null)
  %168 = fpext float %160 to double
  tail call void @__catalyst__qis__RY(double %168, ptr %165, ptr null)
  %169 = fpext float %158 to double
  tail call void @__catalyst__qis__RZ(double %169, ptr %165, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %152, ptr %165, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %165, ptr %140, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %140, ptr %127, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %127, ptr %114, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %114, ptr %101, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %101, ptr %88, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %88, ptr %75, ptr null)
  %170 = fpext float %31 to double
  tail call void @__catalyst__qis__RZ(double %170, ptr %88, ptr null)
  %171 = fpext float %29 to double
  tail call void @__catalyst__qis__RY(double %171, ptr %88, ptr null)
  %172 = fpext float %27 to double
  tail call void @__catalyst__qis__RZ(double %172, ptr %88, ptr null)
  %173 = getelementptr inbounds nuw i8, ptr %1, i64 152
  %174 = load float, ptr %173, align 4, !tbaa !4
  %175 = getelementptr inbounds nuw i8, ptr %1, i64 148
  %176 = load float, ptr %175, align 4, !tbaa !4
  %177 = getelementptr inbounds nuw i8, ptr %1, i64 144
  %178 = load float, ptr %177, align 4, !tbaa !4
  %179 = fpext float %178 to double
  tail call void @__catalyst__qis__RZ(double %179, ptr %114, ptr null)
  %180 = fpext float %176 to double
  tail call void @__catalyst__qis__RY(double %180, ptr %114, ptr null)
  %181 = fpext float %174 to double
  tail call void @__catalyst__qis__RZ(double %181, ptr %114, ptr null)
  %182 = getelementptr inbounds nuw i8, ptr %1, i64 104
  %183 = load float, ptr %182, align 4, !tbaa !4
  %184 = getelementptr inbounds nuw i8, ptr %1, i64 100
  %185 = load float, ptr %184, align 4, !tbaa !4
  %186 = getelementptr inbounds nuw i8, ptr %1, i64 96
  %187 = load float, ptr %186, align 4, !tbaa !4
  tail call void @__catalyst__qis__CNOT(ptr %75, ptr %152, ptr null)
  %188 = fpext float %187 to double
  tail call void @__catalyst__qis__RZ(double %188, ptr %152, ptr null)
  %189 = fpext float %185 to double
  tail call void @__catalyst__qis__RY(double %189, ptr %152, ptr null)
  %190 = fpext float %183 to double
  tail call void @__catalyst__qis__RZ(double %190, ptr %152, ptr null)
  %191 = getelementptr inbounds nuw i8, ptr %1, i64 128
  %192 = load float, ptr %191, align 4, !tbaa !4
  %193 = getelementptr inbounds nuw i8, ptr %1, i64 124
  %194 = load float, ptr %193, align 4, !tbaa !4
  %195 = getelementptr inbounds nuw i8, ptr %1, i64 120
  %196 = load float, ptr %195, align 4, !tbaa !4
  %197 = fpext float %196 to double
  tail call void @__catalyst__qis__RZ(double %197, ptr %140, ptr null)
  %198 = fpext float %194 to double
  tail call void @__catalyst__qis__RY(double %198, ptr %140, ptr null)
  %199 = fpext float %192 to double
  tail call void @__catalyst__qis__RZ(double %199, ptr %140, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %152, ptr %140, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %140, ptr %114, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %114, ptr %88, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %88, ptr %152, ptr null)
  %200 = fpext float %25 to double
  tail call void @__catalyst__qis__RZ(double %200, ptr %152, ptr null)
  %201 = fpext float %23 to double
  tail call void @__catalyst__qis__RY(double %201, ptr %152, ptr null)
  %202 = fpext float %21 to double
  tail call void @__catalyst__qis__RZ(double %202, ptr %152, ptr null)
  %203 = getelementptr inbounds nuw i8, ptr %1, i64 236
  %204 = load float, ptr %203, align 4, !tbaa !4
  %205 = getelementptr inbounds nuw i8, ptr %1, i64 232
  %206 = load float, ptr %205, align 4, !tbaa !4
  %207 = getelementptr inbounds nuw i8, ptr %1, i64 228
  %208 = load float, ptr %207, align 4, !tbaa !4
  %209 = getelementptr inbounds nuw i8, ptr %1, i64 164
  %210 = load float, ptr %209, align 4, !tbaa !4
  %211 = getelementptr inbounds nuw i8, ptr %1, i64 160
  %212 = load float, ptr %211, align 4, !tbaa !4
  %213 = getelementptr inbounds nuw i8, ptr %1, i64 156
  %214 = load float, ptr %213, align 4, !tbaa !4
  %215 = fpext float %214 to double
  tail call void @__catalyst__qis__RZ(double %215, ptr %101, ptr null)
  %216 = fpext float %212 to double
  tail call void @__catalyst__qis__RY(double %216, ptr %101, ptr null)
  %217 = fpext float %210 to double
  tail call void @__catalyst__qis__RZ(double %217, ptr %101, ptr null)
  %218 = getelementptr inbounds nuw i8, ptr %1, i64 116
  %219 = load float, ptr %218, align 4, !tbaa !4
  %220 = getelementptr inbounds nuw i8, ptr %1, i64 112
  %221 = load float, ptr %220, align 4, !tbaa !4
  %222 = getelementptr inbounds nuw i8, ptr %1, i64 108
  %223 = load float, ptr %222, align 4, !tbaa !4
  %224 = fpext float %223 to double
  tail call void @__catalyst__qis__RZ(double %224, ptr %165, ptr null)
  %225 = fpext float %221 to double
  tail call void @__catalyst__qis__RY(double %225, ptr %165, ptr null)
  %226 = fpext float %219 to double
  tail call void @__catalyst__qis__RZ(double %226, ptr %165, ptr null)
  %227 = getelementptr inbounds nuw i8, ptr %1, i64 140
  %228 = load float, ptr %227, align 4, !tbaa !4
  %229 = getelementptr inbounds nuw i8, ptr %1, i64 136
  %230 = load float, ptr %229, align 4, !tbaa !4
  %231 = getelementptr inbounds nuw i8, ptr %1, i64 132
  %232 = load float, ptr %231, align 4, !tbaa !4
  %233 = fpext float %232 to double
  tail call void @__catalyst__qis__RZ(double %233, ptr %127, ptr null)
  %234 = fpext float %230 to double
  tail call void @__catalyst__qis__RY(double %234, ptr %127, ptr null)
  %235 = fpext float %228 to double
  tail call void @__catalyst__qis__RZ(double %235, ptr %127, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %165, ptr %127, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %127, ptr %101, ptr null)
  %236 = fpext float %208 to double
  tail call void @__catalyst__qis__RZ(double %236, ptr %127, ptr null)
  %237 = fpext float %206 to double
  tail call void @__catalyst__qis__RY(double %237, ptr %127, ptr null)
  %238 = fpext float %204 to double
  tail call void @__catalyst__qis__RZ(double %238, ptr %127, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %152, ptr %127, ptr null)
  %239 = getelementptr inbounds nuw i8, ptr %1, i64 224
  %240 = load float, ptr %239, align 4, !tbaa !4
  %241 = getelementptr inbounds nuw i8, ptr %1, i64 220
  %242 = load float, ptr %241, align 4, !tbaa !4
  %243 = getelementptr inbounds nuw i8, ptr %1, i64 216
  %244 = load float, ptr %243, align 4, !tbaa !4
  %245 = fpext float %244 to double
  tail call void @__catalyst__qis__RZ(double %245, ptr %140, ptr null)
  %246 = fpext float %242 to double
  tail call void @__catalyst__qis__RY(double %246, ptr %140, ptr null)
  %247 = fpext float %240 to double
  tail call void @__catalyst__qis__RZ(double %247, ptr %140, ptr null)
  %248 = getelementptr inbounds nuw i8, ptr %1, i64 260
  %249 = load float, ptr %248, align 4, !tbaa !4
  %250 = getelementptr inbounds nuw i8, ptr %1, i64 256
  %251 = load float, ptr %250, align 4, !tbaa !4
  %252 = getelementptr inbounds nuw i8, ptr %1, i64 252
  %253 = load float, ptr %252, align 4, !tbaa !4
  %254 = getelementptr inbounds nuw i8, ptr %1, i64 188
  %255 = load float, ptr %254, align 4, !tbaa !4
  %256 = getelementptr inbounds nuw i8, ptr %1, i64 184
  %257 = load float, ptr %256, align 4, !tbaa !4
  %258 = getelementptr inbounds nuw i8, ptr %1, i64 180
  %259 = load float, ptr %258, align 4, !tbaa !4
  %260 = fpext float %259 to double
  tail call void @__catalyst__qis__RZ(double %260, ptr %75, ptr null)
  %261 = fpext float %257 to double
  tail call void @__catalyst__qis__RY(double %261, ptr %75, ptr null)
  %262 = fpext float %255 to double
  tail call void @__catalyst__qis__RZ(double %262, ptr %75, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %101, ptr %75, ptr null)
  %263 = fpext float %253 to double
  tail call void @__catalyst__qis__RZ(double %263, ptr %101, ptr null)
  %264 = fpext float %251 to double
  tail call void @__catalyst__qis__RY(double %264, ptr %101, ptr null)
  %265 = fpext float %249 to double
  tail call void @__catalyst__qis__RZ(double %265, ptr %101, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %140, ptr %101, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %101, ptr %152, ptr null)
  %266 = fpext float %19 to double
  tail call void @__catalyst__qis__RZ(double %266, ptr %152, ptr null)
  %267 = fpext float %17 to double
  tail call void @__catalyst__qis__RY(double %267, ptr %152, ptr null)
  %268 = fpext float %15 to double
  tail call void @__catalyst__qis__RZ(double %268, ptr %152, ptr null)
  %269 = getelementptr inbounds nuw i8, ptr %1, i64 344
  %270 = load float, ptr %269, align 4, !tbaa !4
  %271 = getelementptr inbounds nuw i8, ptr %1, i64 340
  %272 = load float, ptr %271, align 4, !tbaa !4
  %273 = getelementptr inbounds nuw i8, ptr %1, i64 336
  %274 = load float, ptr %273, align 4, !tbaa !4
  %275 = getelementptr inbounds nuw i8, ptr %1, i64 284
  %276 = load float, ptr %275, align 4, !tbaa !4
  %277 = getelementptr inbounds nuw i8, ptr %1, i64 280
  %278 = load float, ptr %277, align 4, !tbaa !4
  %279 = getelementptr inbounds nuw i8, ptr %1, i64 276
  %280 = load float, ptr %279, align 4, !tbaa !4
  tail call void @__catalyst__qis__CNOT(ptr %75, ptr %165, ptr null)
  %281 = fpext float %280 to double
  tail call void @__catalyst__qis__RZ(double %281, ptr %75, ptr null)
  %282 = fpext float %278 to double
  tail call void @__catalyst__qis__RY(double %282, ptr %75, ptr null)
  %283 = fpext float %276 to double
  tail call void @__catalyst__qis__RZ(double %283, ptr %75, ptr null)
  %284 = getelementptr inbounds nuw i8, ptr %1, i64 212
  %285 = load float, ptr %284, align 4, !tbaa !4
  %286 = getelementptr inbounds nuw i8, ptr %1, i64 208
  %287 = load float, ptr %286, align 4, !tbaa !4
  %288 = getelementptr inbounds nuw i8, ptr %1, i64 204
  %289 = load float, ptr %288, align 4, !tbaa !4
  %290 = fpext float %289 to double
  tail call void @__catalyst__qis__RZ(double %290, ptr %165, ptr null)
  %291 = fpext float %287 to double
  tail call void @__catalyst__qis__RY(double %291, ptr %165, ptr null)
  %292 = fpext float %285 to double
  tail call void @__catalyst__qis__RZ(double %292, ptr %165, ptr null)
  %293 = getelementptr inbounds nuw i8, ptr %1, i64 248
  %294 = load float, ptr %293, align 4, !tbaa !4
  %295 = getelementptr inbounds nuw i8, ptr %1, i64 244
  %296 = load float, ptr %295, align 4, !tbaa !4
  %297 = getelementptr inbounds nuw i8, ptr %1, i64 240
  %298 = load float, ptr %297, align 4, !tbaa !4
  %299 = fpext float %298 to double
  tail call void @__catalyst__qis__RZ(double %299, ptr %114, ptr null)
  %300 = fpext float %296 to double
  tail call void @__catalyst__qis__RY(double %300, ptr %114, ptr null)
  %301 = fpext float %294 to double
  tail call void @__catalyst__qis__RZ(double %301, ptr %114, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %165, ptr %114, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %114, ptr %75, ptr null)
  %302 = fpext float %274 to double
  tail call void @__catalyst__qis__RZ(double %302, ptr %114, ptr null)
  %303 = fpext float %272 to double
  tail call void @__catalyst__qis__RY(double %303, ptr %114, ptr null)
  %304 = fpext float %270 to double
  tail call void @__catalyst__qis__RZ(double %304, ptr %114, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %152, ptr %114, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %114, ptr %152, ptr null)
  %305 = getelementptr inbounds nuw i8, ptr %1, i64 308
  %306 = load float, ptr %305, align 4, !tbaa !4
  %307 = getelementptr inbounds nuw i8, ptr %1, i64 304
  %308 = load float, ptr %307, align 4, !tbaa !4
  %309 = getelementptr inbounds nuw i8, ptr %1, i64 300
  %310 = load float, ptr %309, align 4, !tbaa !4
  %311 = getelementptr inbounds nuw i8, ptr %1, i64 272
  %312 = load float, ptr %311, align 4, !tbaa !4
  %313 = getelementptr inbounds nuw i8, ptr %1, i64 268
  %314 = load float, ptr %313, align 4, !tbaa !4
  %315 = getelementptr inbounds nuw i8, ptr %1, i64 264
  %316 = load float, ptr %315, align 4, !tbaa !4
  %317 = fpext float %316 to double
  tail call void @__catalyst__qis__RZ(double %317, ptr %88, ptr null)
  %318 = fpext float %314 to double
  tail call void @__catalyst__qis__RY(double %318, ptr %88, ptr null)
  %319 = fpext float %312 to double
  tail call void @__catalyst__qis__RZ(double %319, ptr %88, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %127, ptr %88, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %88, ptr %165, ptr null)
  %320 = fpext float %310 to double
  tail call void @__catalyst__qis__RZ(double %320, ptr %165, ptr null)
  %321 = fpext float %308 to double
  tail call void @__catalyst__qis__RY(double %321, ptr %165, ptr null)
  %322 = fpext float %306 to double
  tail call void @__catalyst__qis__RZ(double %322, ptr %165, ptr null)
  %323 = getelementptr inbounds nuw i8, ptr %1, i64 356
  %324 = load float, ptr %323, align 4, !tbaa !4
  %325 = getelementptr inbounds nuw i8, ptr %1, i64 352
  %326 = load float, ptr %325, align 4, !tbaa !4
  %327 = getelementptr inbounds nuw i8, ptr %1, i64 348
  %328 = load float, ptr %327, align 4, !tbaa !4
  %329 = fpext float %328 to double
  tail call void @__catalyst__qis__RZ(double %329, ptr %101, ptr null)
  %330 = fpext float %326 to double
  tail call void @__catalyst__qis__RY(double %330, ptr %101, ptr null)
  %331 = fpext float %324 to double
  tail call void @__catalyst__qis__RZ(double %331, ptr %101, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %165, ptr %101, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %101, ptr %165, ptr null)
  %332 = getelementptr inbounds nuw i8, ptr %1, i64 320
  %333 = load float, ptr %332, align 4, !tbaa !4
  %334 = getelementptr inbounds nuw i8, ptr %1, i64 316
  %335 = load float, ptr %334, align 4, !tbaa !4
  %336 = getelementptr inbounds nuw i8, ptr %1, i64 312
  %337 = load float, ptr %336, align 4, !tbaa !4
  tail call void @__catalyst__qis__CNOT(ptr %75, ptr %140, ptr null)
  %338 = fpext float %337 to double
  tail call void @__catalyst__qis__RZ(double %338, ptr %140, ptr null)
  %339 = fpext float %335 to double
  tail call void @__catalyst__qis__RY(double %339, ptr %140, ptr null)
  %340 = fpext float %333 to double
  tail call void @__catalyst__qis__RZ(double %340, ptr %140, ptr null)
  %341 = getelementptr inbounds nuw i8, ptr %1, i64 368
  %342 = load float, ptr %341, align 4, !tbaa !4
  %343 = getelementptr inbounds nuw i8, ptr %1, i64 364
  %344 = load float, ptr %343, align 4, !tbaa !4
  %345 = getelementptr inbounds nuw i8, ptr %1, i64 360
  %346 = load float, ptr %345, align 4, !tbaa !4
  %347 = fpext float %346 to double
  tail call void @__catalyst__qis__RZ(double %347, ptr %88, ptr null)
  %348 = fpext float %344 to double
  tail call void @__catalyst__qis__RY(double %348, ptr %88, ptr null)
  %349 = fpext float %342 to double
  tail call void @__catalyst__qis__RZ(double %349, ptr %88, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %140, ptr %88, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %88, ptr %140, ptr null)
  %350 = getelementptr inbounds nuw i8, ptr %1, i64 332
  %351 = load float, ptr %350, align 4, !tbaa !4
  %352 = getelementptr inbounds nuw i8, ptr %1, i64 328
  %353 = load float, ptr %352, align 4, !tbaa !4
  %354 = getelementptr inbounds nuw i8, ptr %1, i64 324
  %355 = load float, ptr %354, align 4, !tbaa !4
  %356 = fpext float %355 to double
  tail call void @__catalyst__qis__RZ(double %356, ptr %127, ptr null)
  %357 = fpext float %353 to double
  tail call void @__catalyst__qis__RY(double %357, ptr %127, ptr null)
  %358 = fpext float %351 to double
  tail call void @__catalyst__qis__RZ(double %358, ptr %127, ptr null)
  %359 = getelementptr inbounds nuw i8, ptr %1, i64 380
  %360 = load float, ptr %359, align 4, !tbaa !4
  %361 = getelementptr inbounds nuw i8, ptr %1, i64 376
  %362 = load float, ptr %361, align 4, !tbaa !4
  %363 = getelementptr inbounds nuw i8, ptr %1, i64 372
  %364 = load float, ptr %363, align 4, !tbaa !4
  %365 = fpext float %364 to double
  tail call void @__catalyst__qis__RZ(double %365, ptr %75, ptr null)
  %366 = fpext float %362 to double
  tail call void @__catalyst__qis__RY(double %366, ptr %75, ptr null)
  %367 = fpext float %360 to double
  tail call void @__catalyst__qis__RZ(double %367, ptr %75, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %127, ptr %75, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %75, ptr %127, ptr null)
  %368 = tail call i64 @__catalyst__qis__NamedObs(i64 3, ptr %152)
  %369 = tail call double @__catalyst__qis__Expval(i64 %368)
  %370 = insertvalue { ptr, double } poison, ptr %73, 0
  %371 = insertvalue { ptr, double } %370, double %369, 1
  ret { ptr, double } %371
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none)
define noundef i64 @qnode_forward_0.pcount(ptr readnone captures(none) %0, ptr readnone captures(none) %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, ptr readnone captures(none) %9, ptr readnone captures(none) %10, i64 %11, i64 %12, i64 %13) local_unnamed_addr #0 {
  ret i64 104
}

define void @qnode_forward_0.quantum.customqgrad(ptr readonly captures(none) %0, ptr readnone captures(none) %1, ptr readonly captures(none) %2, ptr readnone captures(none) %3, ptr readnone captures(none) %4, ptr readonly captures(none) %5, ptr readnone captures(none) %6, ptr readonly captures(none) %7, ptr readnone captures(none) %8) {
  %10 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, align 8
  %.elt3 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %.unpack4 = load ptr, ptr %.elt3, align 8
  %.elt22 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %.unpack23 = load ptr, ptr %.elt22, align 8
  %.elt33 = getelementptr inbounds nuw i8, ptr %5, i64 8
  %.unpack34 = load ptr, ptr %.elt33, align 8
  %.elt37 = getelementptr inbounds nuw i8, ptr %5, i64 24
  %.unpack38.unpack = load i64, ptr %.elt37, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %10)
  %.idx.i = shl i64 %.unpack38.unpack, 3
  %11 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 %.idx.i)
  tail call void @__catalyst__rt__toggle_recorder(i1 true)
  %12 = tail call { ptr, double } @qnode_forward_0.nodealloc(ptr readnone poison, ptr %.unpack4, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, ptr readnone poison, ptr readonly %.unpack23, i64 poison, i64 poison, i64 poison)
  %13 = extractvalue { ptr, double } %12, 0
  tail call void @__catalyst__rt__toggle_recorder(i1 false)
  store ptr %11, ptr %10, align 8
  %.fca.1.gep.i = getelementptr inbounds nuw i8, ptr %10, i64 8
  store ptr %11, ptr %.fca.1.gep.i, align 8
  %.fca.2.gep.i = getelementptr inbounds nuw i8, ptr %10, i64 16
  store i64 0, ptr %.fca.2.gep.i, align 8
  %.fca.3.0.gep.i = getelementptr inbounds nuw i8, ptr %10, i64 24
  store i64 %.unpack38.unpack, ptr %.fca.3.0.gep.i, align 8
  %.fca.4.0.gep.i = getelementptr inbounds nuw i8, ptr %10, i64 32
  store i64 1, ptr %.fca.4.0.gep.i, align 8
  call void (i64, ...) @__catalyst__qis__Gradient(i64 1, ptr nonnull %10)
  call void @__catalyst__rt__qubit_release_array(ptr %13)
  call void @__catalyst__rt__device_release()
  call void @llvm.lifetime.end.p0(ptr nonnull %10)
  %.elt44 = getelementptr inbounds nuw i8, ptr %7, i64 8
  %.unpack45 = load ptr, ptr %.elt44, align 8
  %14 = icmp sgt i64 %.unpack38.unpack, 0
  br i1 %14, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %9, %.lr.ph
  %15 = phi i64 [ %23, %.lr.ph ], [ 0, %9 ]
  %16 = load double, ptr %.unpack45, align 8, !tbaa !1
  %17 = getelementptr inbounds nuw double, ptr %11, i64 %15
  %18 = load double, ptr %17, align 8, !tbaa !1
  %19 = getelementptr inbounds nuw double, ptr %.unpack34, i64 %15
  %20 = load double, ptr %19, align 8, !tbaa !1
  %21 = fmul double %16, %18
  %22 = fadd double %20, %21
  store double %22, ptr %19, align 8, !tbaa !1
  %23 = add nuw nsw i64 %15, 1
  %exitcond.not = icmp eq i64 %23, %.unpack38.unpack
  br i1 %exitcond.not, label %._crit_edge, label %.lr.ph

._crit_edge:                                      ; preds = %.lr.ph, %9
  ret void
}

; Function Attrs: noinline
define void @qnode_forward_0.quantum(ptr %0, ptr %1, ptr %2, ptr %3) #1 {
  %5 = load volatile { ptr, ptr, i64, [3 x i64], [3 x i64] }, ptr %0, align 8
  %6 = load volatile { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %1, align 8
  %7 = load volatile { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %2, align 8
  %8 = load volatile { ptr, ptr, i64 }, ptr %3, align 8
  tail call void @__catalyst__rt__device_init(ptr nonnull @"/home/pichau/QAgents/.venv/lib/python3.12/site-packages/pennylane_lightning/liblightning_gpu_catalyst.so", ptr nonnull @LightningGPUSimulator, ptr nonnull @"{}", i64 0, i1 false)
  %9 = tail call ptr @__catalyst__rt__qubit_allocate_array(i64 8)
  %10 = tail call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %9, i64 7)
  %11 = load ptr, ptr %10, align 8
  %12 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %13 = load double, ptr %12, align 8, !tbaa !1
  tail call void @__catalyst__qis__RY(double %13, ptr %11, ptr null)
  %14 = getelementptr inbounds nuw i8, ptr %12, i64 8
  %15 = load double, ptr %14, align 8, !tbaa !1
  tail call void @__catalyst__qis__RZ(double %15, ptr %11, ptr null)
  %16 = getelementptr inbounds nuw i8, ptr %12, i64 16
  %17 = load double, ptr %16, align 8, !tbaa !1
  tail call void @__catalyst__qis__RY(double %17, ptr %11, ptr null)
  %18 = getelementptr inbounds nuw i8, ptr %12, i64 24
  %19 = load double, ptr %18, align 8, !tbaa !1
  tail call void @__catalyst__qis__RZ(double %19, ptr %11, ptr null)
  %20 = tail call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %9, i64 6)
  %21 = load ptr, ptr %20, align 8
  %22 = getelementptr inbounds nuw i8, ptr %12, i64 32
  %23 = load double, ptr %22, align 8, !tbaa !1
  tail call void @__catalyst__qis__RY(double %23, ptr %21, ptr null)
  %24 = getelementptr inbounds nuw i8, ptr %12, i64 40
  %25 = load double, ptr %24, align 8, !tbaa !1
  tail call void @__catalyst__qis__RZ(double %25, ptr %21, ptr null)
  %26 = getelementptr inbounds nuw i8, ptr %12, i64 48
  %27 = load double, ptr %26, align 8, !tbaa !1
  tail call void @__catalyst__qis__RY(double %27, ptr %21, ptr null)
  %28 = getelementptr inbounds nuw i8, ptr %12, i64 56
  %29 = load double, ptr %28, align 8, !tbaa !1
  tail call void @__catalyst__qis__RZ(double %29, ptr %21, ptr null)
  %30 = tail call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %9, i64 5)
  %31 = load ptr, ptr %30, align 8
  %32 = getelementptr inbounds nuw i8, ptr %12, i64 64
  %33 = load double, ptr %32, align 8, !tbaa !1
  tail call void @__catalyst__qis__RY(double %33, ptr %31, ptr null)
  %34 = getelementptr inbounds nuw i8, ptr %12, i64 72
  %35 = load double, ptr %34, align 8, !tbaa !1
  tail call void @__catalyst__qis__RZ(double %35, ptr %31, ptr null)
  %36 = getelementptr inbounds nuw i8, ptr %12, i64 80
  %37 = load double, ptr %36, align 8, !tbaa !1
  tail call void @__catalyst__qis__RY(double %37, ptr %31, ptr null)
  %38 = getelementptr inbounds nuw i8, ptr %12, i64 88
  %39 = load double, ptr %38, align 8, !tbaa !1
  tail call void @__catalyst__qis__RZ(double %39, ptr %31, ptr null)
  %40 = tail call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %9, i64 4)
  %41 = load ptr, ptr %40, align 8
  %42 = getelementptr inbounds nuw i8, ptr %12, i64 96
  %43 = load double, ptr %42, align 8, !tbaa !1
  tail call void @__catalyst__qis__RY(double %43, ptr %41, ptr null)
  %44 = getelementptr inbounds nuw i8, ptr %12, i64 104
  %45 = load double, ptr %44, align 8, !tbaa !1
  tail call void @__catalyst__qis__RZ(double %45, ptr %41, ptr null)
  %46 = getelementptr inbounds nuw i8, ptr %12, i64 112
  %47 = load double, ptr %46, align 8, !tbaa !1
  tail call void @__catalyst__qis__RY(double %47, ptr %41, ptr null)
  %48 = getelementptr inbounds nuw i8, ptr %12, i64 120
  %49 = load double, ptr %48, align 8, !tbaa !1
  tail call void @__catalyst__qis__RZ(double %49, ptr %41, ptr null)
  %50 = tail call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %9, i64 3)
  %51 = load ptr, ptr %50, align 8
  %52 = getelementptr inbounds nuw i8, ptr %12, i64 128
  %53 = load double, ptr %52, align 8, !tbaa !1
  tail call void @__catalyst__qis__RY(double %53, ptr %51, ptr null)
  %54 = getelementptr inbounds nuw i8, ptr %12, i64 136
  %55 = load double, ptr %54, align 8, !tbaa !1
  tail call void @__catalyst__qis__RZ(double %55, ptr %51, ptr null)
  %56 = getelementptr inbounds nuw i8, ptr %12, i64 144
  %57 = load double, ptr %56, align 8, !tbaa !1
  tail call void @__catalyst__qis__RY(double %57, ptr %51, ptr null)
  %58 = getelementptr inbounds nuw i8, ptr %12, i64 152
  %59 = load double, ptr %58, align 8, !tbaa !1
  tail call void @__catalyst__qis__RZ(double %59, ptr %51, ptr null)
  %60 = tail call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %9, i64 2)
  %61 = load ptr, ptr %60, align 8
  %62 = getelementptr inbounds nuw i8, ptr %12, i64 160
  %63 = load double, ptr %62, align 8, !tbaa !1
  tail call void @__catalyst__qis__RY(double %63, ptr %61, ptr null)
  %64 = getelementptr inbounds nuw i8, ptr %12, i64 168
  %65 = load double, ptr %64, align 8, !tbaa !1
  tail call void @__catalyst__qis__RZ(double %65, ptr %61, ptr null)
  %66 = getelementptr inbounds nuw i8, ptr %12, i64 176
  %67 = load double, ptr %66, align 8, !tbaa !1
  tail call void @__catalyst__qis__RY(double %67, ptr %61, ptr null)
  %68 = getelementptr inbounds nuw i8, ptr %12, i64 184
  %69 = load double, ptr %68, align 8, !tbaa !1
  tail call void @__catalyst__qis__RZ(double %69, ptr %61, ptr null)
  %70 = tail call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %9, i64 0)
  %71 = load ptr, ptr %70, align 8
  %72 = getelementptr inbounds nuw i8, ptr %12, i64 192
  %73 = load double, ptr %72, align 8, !tbaa !1
  tail call void @__catalyst__qis__RY(double %73, ptr %71, ptr null)
  %74 = getelementptr inbounds nuw i8, ptr %12, i64 200
  %75 = load double, ptr %74, align 8, !tbaa !1
  tail call void @__catalyst__qis__RZ(double %75, ptr %71, ptr null)
  %76 = getelementptr inbounds nuw i8, ptr %12, i64 208
  %77 = load double, ptr %76, align 8, !tbaa !1
  tail call void @__catalyst__qis__RY(double %77, ptr %71, ptr null)
  %78 = getelementptr inbounds nuw i8, ptr %12, i64 216
  %79 = load double, ptr %78, align 8, !tbaa !1
  tail call void @__catalyst__qis__RZ(double %79, ptr %71, ptr null)
  %80 = tail call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %9, i64 1)
  %81 = load ptr, ptr %80, align 8
  %82 = getelementptr inbounds nuw i8, ptr %12, i64 224
  %83 = load double, ptr %82, align 8, !tbaa !1
  tail call void @__catalyst__qis__RY(double %83, ptr %81, ptr null)
  %84 = getelementptr inbounds nuw i8, ptr %12, i64 232
  %85 = load double, ptr %84, align 8, !tbaa !1
  tail call void @__catalyst__qis__RZ(double %85, ptr %81, ptr null)
  %86 = getelementptr inbounds nuw i8, ptr %12, i64 240
  %87 = load double, ptr %86, align 8, !tbaa !1
  tail call void @__catalyst__qis__RY(double %87, ptr %81, ptr null)
  %88 = getelementptr inbounds nuw i8, ptr %12, i64 248
  %89 = load double, ptr %88, align 8, !tbaa !1
  tail call void @__catalyst__qis__RZ(double %89, ptr %81, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %71, ptr %81, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %81, ptr %61, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %61, ptr %51, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %51, ptr %41, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %41, ptr %31, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %31, ptr %21, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %21, ptr %11, ptr null)
  %90 = getelementptr inbounds nuw i8, ptr %12, i64 256
  %91 = load double, ptr %90, align 8, !tbaa !1
  tail call void @__catalyst__qis__RZ(double %91, ptr %21, ptr null)
  %92 = getelementptr inbounds nuw i8, ptr %12, i64 264
  %93 = load double, ptr %92, align 8, !tbaa !1
  tail call void @__catalyst__qis__RY(double %93, ptr %21, ptr null)
  %94 = getelementptr inbounds nuw i8, ptr %12, i64 272
  %95 = load double, ptr %94, align 8, !tbaa !1
  tail call void @__catalyst__qis__RZ(double %95, ptr %21, ptr null)
  %96 = getelementptr inbounds nuw i8, ptr %12, i64 280
  %97 = load double, ptr %96, align 8, !tbaa !1
  tail call void @__catalyst__qis__RZ(double %97, ptr %41, ptr null)
  %98 = getelementptr inbounds nuw i8, ptr %12, i64 288
  %99 = load double, ptr %98, align 8, !tbaa !1
  tail call void @__catalyst__qis__RY(double %99, ptr %41, ptr null)
  %100 = getelementptr inbounds nuw i8, ptr %12, i64 296
  %101 = load double, ptr %100, align 8, !tbaa !1
  tail call void @__catalyst__qis__RZ(double %101, ptr %41, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %11, ptr %71, ptr null)
  %102 = getelementptr inbounds nuw i8, ptr %12, i64 304
  %103 = load double, ptr %102, align 8, !tbaa !1
  tail call void @__catalyst__qis__RZ(double %103, ptr %71, ptr null)
  %104 = getelementptr inbounds nuw i8, ptr %12, i64 312
  %105 = load double, ptr %104, align 8, !tbaa !1
  tail call void @__catalyst__qis__RY(double %105, ptr %71, ptr null)
  %106 = getelementptr inbounds nuw i8, ptr %12, i64 320
  %107 = load double, ptr %106, align 8, !tbaa !1
  tail call void @__catalyst__qis__RZ(double %107, ptr %71, ptr null)
  %108 = getelementptr inbounds nuw i8, ptr %12, i64 328
  %109 = load double, ptr %108, align 8, !tbaa !1
  tail call void @__catalyst__qis__RZ(double %109, ptr %61, ptr null)
  %110 = getelementptr inbounds nuw i8, ptr %12, i64 336
  %111 = load double, ptr %110, align 8, !tbaa !1
  tail call void @__catalyst__qis__RY(double %111, ptr %61, ptr null)
  %112 = getelementptr inbounds nuw i8, ptr %12, i64 344
  %113 = load double, ptr %112, align 8, !tbaa !1
  tail call void @__catalyst__qis__RZ(double %113, ptr %61, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %71, ptr %61, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %61, ptr %41, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %41, ptr %21, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %21, ptr %71, ptr null)
  %114 = getelementptr inbounds nuw i8, ptr %12, i64 352
  %115 = load double, ptr %114, align 8, !tbaa !1
  tail call void @__catalyst__qis__RZ(double %115, ptr %71, ptr null)
  %116 = getelementptr inbounds nuw i8, ptr %12, i64 360
  %117 = load double, ptr %116, align 8, !tbaa !1
  tail call void @__catalyst__qis__RY(double %117, ptr %71, ptr null)
  %118 = getelementptr inbounds nuw i8, ptr %12, i64 368
  %119 = load double, ptr %118, align 8, !tbaa !1
  tail call void @__catalyst__qis__RZ(double %119, ptr %71, ptr null)
  %120 = getelementptr inbounds nuw i8, ptr %12, i64 376
  %121 = load double, ptr %120, align 8, !tbaa !1
  tail call void @__catalyst__qis__RZ(double %121, ptr %31, ptr null)
  %122 = getelementptr inbounds nuw i8, ptr %12, i64 384
  %123 = load double, ptr %122, align 8, !tbaa !1
  tail call void @__catalyst__qis__RY(double %123, ptr %31, ptr null)
  %124 = getelementptr inbounds nuw i8, ptr %12, i64 392
  %125 = load double, ptr %124, align 8, !tbaa !1
  tail call void @__catalyst__qis__RZ(double %125, ptr %31, ptr null)
  %126 = getelementptr inbounds nuw i8, ptr %12, i64 400
  %127 = load double, ptr %126, align 8, !tbaa !1
  tail call void @__catalyst__qis__RZ(double %127, ptr %81, ptr null)
  %128 = getelementptr inbounds nuw i8, ptr %12, i64 408
  %129 = load double, ptr %128, align 8, !tbaa !1
  tail call void @__catalyst__qis__RY(double %129, ptr %81, ptr null)
  %130 = getelementptr inbounds nuw i8, ptr %12, i64 416
  %131 = load double, ptr %130, align 8, !tbaa !1
  tail call void @__catalyst__qis__RZ(double %131, ptr %81, ptr null)
  %132 = getelementptr inbounds nuw i8, ptr %12, i64 424
  %133 = load double, ptr %132, align 8, !tbaa !1
  tail call void @__catalyst__qis__RZ(double %133, ptr %51, ptr null)
  %134 = getelementptr inbounds nuw i8, ptr %12, i64 432
  %135 = load double, ptr %134, align 8, !tbaa !1
  tail call void @__catalyst__qis__RY(double %135, ptr %51, ptr null)
  %136 = getelementptr inbounds nuw i8, ptr %12, i64 440
  %137 = load double, ptr %136, align 8, !tbaa !1
  tail call void @__catalyst__qis__RZ(double %137, ptr %51, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %81, ptr %51, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %51, ptr %31, ptr null)
  %138 = getelementptr inbounds nuw i8, ptr %12, i64 448
  %139 = load double, ptr %138, align 8, !tbaa !1
  tail call void @__catalyst__qis__RZ(double %139, ptr %51, ptr null)
  %140 = getelementptr inbounds nuw i8, ptr %12, i64 456
  %141 = load double, ptr %140, align 8, !tbaa !1
  tail call void @__catalyst__qis__RY(double %141, ptr %51, ptr null)
  %142 = getelementptr inbounds nuw i8, ptr %12, i64 464
  %143 = load double, ptr %142, align 8, !tbaa !1
  tail call void @__catalyst__qis__RZ(double %143, ptr %51, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %71, ptr %51, ptr null)
  %144 = getelementptr inbounds nuw i8, ptr %12, i64 472
  %145 = load double, ptr %144, align 8, !tbaa !1
  tail call void @__catalyst__qis__RZ(double %145, ptr %61, ptr null)
  %146 = getelementptr inbounds nuw i8, ptr %12, i64 480
  %147 = load double, ptr %146, align 8, !tbaa !1
  tail call void @__catalyst__qis__RY(double %147, ptr %61, ptr null)
  %148 = getelementptr inbounds nuw i8, ptr %12, i64 488
  %149 = load double, ptr %148, align 8, !tbaa !1
  tail call void @__catalyst__qis__RZ(double %149, ptr %61, ptr null)
  %150 = getelementptr inbounds nuw i8, ptr %12, i64 496
  %151 = load double, ptr %150, align 8, !tbaa !1
  tail call void @__catalyst__qis__RZ(double %151, ptr %11, ptr null)
  %152 = getelementptr inbounds nuw i8, ptr %12, i64 504
  %153 = load double, ptr %152, align 8, !tbaa !1
  tail call void @__catalyst__qis__RY(double %153, ptr %11, ptr null)
  %154 = getelementptr inbounds nuw i8, ptr %12, i64 512
  %155 = load double, ptr %154, align 8, !tbaa !1
  tail call void @__catalyst__qis__RZ(double %155, ptr %11, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %31, ptr %11, ptr null)
  %156 = getelementptr inbounds nuw i8, ptr %12, i64 520
  %157 = load double, ptr %156, align 8, !tbaa !1
  tail call void @__catalyst__qis__RZ(double %157, ptr %31, ptr null)
  %158 = getelementptr inbounds nuw i8, ptr %12, i64 528
  %159 = load double, ptr %158, align 8, !tbaa !1
  tail call void @__catalyst__qis__RY(double %159, ptr %31, ptr null)
  %160 = getelementptr inbounds nuw i8, ptr %12, i64 536
  %161 = load double, ptr %160, align 8, !tbaa !1
  tail call void @__catalyst__qis__RZ(double %161, ptr %31, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %61, ptr %31, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %31, ptr %71, ptr null)
  %162 = getelementptr inbounds nuw i8, ptr %12, i64 544
  %163 = load double, ptr %162, align 8, !tbaa !1
  tail call void @__catalyst__qis__RZ(double %163, ptr %71, ptr null)
  %164 = getelementptr inbounds nuw i8, ptr %12, i64 552
  %165 = load double, ptr %164, align 8, !tbaa !1
  tail call void @__catalyst__qis__RY(double %165, ptr %71, ptr null)
  %166 = getelementptr inbounds nuw i8, ptr %12, i64 560
  %167 = load double, ptr %166, align 8, !tbaa !1
  tail call void @__catalyst__qis__RZ(double %167, ptr %71, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %11, ptr %81, ptr null)
  %168 = getelementptr inbounds nuw i8, ptr %12, i64 568
  %169 = load double, ptr %168, align 8, !tbaa !1
  tail call void @__catalyst__qis__RZ(double %169, ptr %11, ptr null)
  %170 = getelementptr inbounds nuw i8, ptr %12, i64 576
  %171 = load double, ptr %170, align 8, !tbaa !1
  tail call void @__catalyst__qis__RY(double %171, ptr %11, ptr null)
  %172 = getelementptr inbounds nuw i8, ptr %12, i64 584
  %173 = load double, ptr %172, align 8, !tbaa !1
  tail call void @__catalyst__qis__RZ(double %173, ptr %11, ptr null)
  %174 = getelementptr inbounds nuw i8, ptr %12, i64 592
  %175 = load double, ptr %174, align 8, !tbaa !1
  tail call void @__catalyst__qis__RZ(double %175, ptr %81, ptr null)
  %176 = getelementptr inbounds nuw i8, ptr %12, i64 600
  %177 = load double, ptr %176, align 8, !tbaa !1
  tail call void @__catalyst__qis__RY(double %177, ptr %81, ptr null)
  %178 = getelementptr inbounds nuw i8, ptr %12, i64 608
  %179 = load double, ptr %178, align 8, !tbaa !1
  tail call void @__catalyst__qis__RZ(double %179, ptr %81, ptr null)
  %180 = getelementptr inbounds nuw i8, ptr %12, i64 616
  %181 = load double, ptr %180, align 8, !tbaa !1
  tail call void @__catalyst__qis__RZ(double %181, ptr %41, ptr null)
  %182 = getelementptr inbounds nuw i8, ptr %12, i64 624
  %183 = load double, ptr %182, align 8, !tbaa !1
  tail call void @__catalyst__qis__RY(double %183, ptr %41, ptr null)
  %184 = getelementptr inbounds nuw i8, ptr %12, i64 632
  %185 = load double, ptr %184, align 8, !tbaa !1
  tail call void @__catalyst__qis__RZ(double %185, ptr %41, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %81, ptr %41, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %41, ptr %11, ptr null)
  %186 = getelementptr inbounds nuw i8, ptr %12, i64 640
  %187 = load double, ptr %186, align 8, !tbaa !1
  tail call void @__catalyst__qis__RZ(double %187, ptr %41, ptr null)
  %188 = getelementptr inbounds nuw i8, ptr %12, i64 648
  %189 = load double, ptr %188, align 8, !tbaa !1
  tail call void @__catalyst__qis__RY(double %189, ptr %41, ptr null)
  %190 = getelementptr inbounds nuw i8, ptr %12, i64 656
  %191 = load double, ptr %190, align 8, !tbaa !1
  tail call void @__catalyst__qis__RZ(double %191, ptr %41, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %71, ptr %41, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %41, ptr %71, ptr null)
  %192 = getelementptr inbounds nuw i8, ptr %12, i64 664
  %193 = load double, ptr %192, align 8, !tbaa !1
  tail call void @__catalyst__qis__RZ(double %193, ptr %21, ptr null)
  %194 = getelementptr inbounds nuw i8, ptr %12, i64 672
  %195 = load double, ptr %194, align 8, !tbaa !1
  tail call void @__catalyst__qis__RY(double %195, ptr %21, ptr null)
  %196 = getelementptr inbounds nuw i8, ptr %12, i64 680
  %197 = load double, ptr %196, align 8, !tbaa !1
  tail call void @__catalyst__qis__RZ(double %197, ptr %21, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %51, ptr %21, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %21, ptr %81, ptr null)
  %198 = getelementptr inbounds nuw i8, ptr %12, i64 688
  %199 = load double, ptr %198, align 8, !tbaa !1
  tail call void @__catalyst__qis__RZ(double %199, ptr %81, ptr null)
  %200 = getelementptr inbounds nuw i8, ptr %12, i64 696
  %201 = load double, ptr %200, align 8, !tbaa !1
  tail call void @__catalyst__qis__RY(double %201, ptr %81, ptr null)
  %202 = getelementptr inbounds nuw i8, ptr %12, i64 704
  %203 = load double, ptr %202, align 8, !tbaa !1
  tail call void @__catalyst__qis__RZ(double %203, ptr %81, ptr null)
  %204 = getelementptr inbounds nuw i8, ptr %12, i64 712
  %205 = load double, ptr %204, align 8, !tbaa !1
  tail call void @__catalyst__qis__RZ(double %205, ptr %31, ptr null)
  %206 = getelementptr inbounds nuw i8, ptr %12, i64 720
  %207 = load double, ptr %206, align 8, !tbaa !1
  tail call void @__catalyst__qis__RY(double %207, ptr %31, ptr null)
  %208 = getelementptr inbounds nuw i8, ptr %12, i64 728
  %209 = load double, ptr %208, align 8, !tbaa !1
  tail call void @__catalyst__qis__RZ(double %209, ptr %31, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %81, ptr %31, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %31, ptr %81, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %11, ptr %61, ptr null)
  %210 = getelementptr inbounds nuw i8, ptr %12, i64 736
  %211 = load double, ptr %210, align 8, !tbaa !1
  tail call void @__catalyst__qis__RZ(double %211, ptr %61, ptr null)
  %212 = getelementptr inbounds nuw i8, ptr %12, i64 744
  %213 = load double, ptr %212, align 8, !tbaa !1
  tail call void @__catalyst__qis__RY(double %213, ptr %61, ptr null)
  %214 = getelementptr inbounds nuw i8, ptr %12, i64 752
  %215 = load double, ptr %214, align 8, !tbaa !1
  tail call void @__catalyst__qis__RZ(double %215, ptr %61, ptr null)
  %216 = getelementptr inbounds nuw i8, ptr %12, i64 760
  %217 = load double, ptr %216, align 8, !tbaa !1
  tail call void @__catalyst__qis__RZ(double %217, ptr %21, ptr null)
  %218 = getelementptr inbounds nuw i8, ptr %12, i64 768
  %219 = load double, ptr %218, align 8, !tbaa !1
  tail call void @__catalyst__qis__RY(double %219, ptr %21, ptr null)
  %220 = getelementptr inbounds nuw i8, ptr %12, i64 776
  %221 = load double, ptr %220, align 8, !tbaa !1
  tail call void @__catalyst__qis__RZ(double %221, ptr %21, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %61, ptr %21, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %21, ptr %61, ptr null)
  %222 = getelementptr inbounds nuw i8, ptr %12, i64 784
  %223 = load double, ptr %222, align 8, !tbaa !1
  tail call void @__catalyst__qis__RZ(double %223, ptr %51, ptr null)
  %224 = getelementptr inbounds nuw i8, ptr %12, i64 792
  %225 = load double, ptr %224, align 8, !tbaa !1
  tail call void @__catalyst__qis__RY(double %225, ptr %51, ptr null)
  %226 = getelementptr inbounds nuw i8, ptr %12, i64 800
  %227 = load double, ptr %226, align 8, !tbaa !1
  tail call void @__catalyst__qis__RZ(double %227, ptr %51, ptr null)
  %228 = getelementptr inbounds nuw i8, ptr %12, i64 808
  %229 = load double, ptr %228, align 8, !tbaa !1
  tail call void @__catalyst__qis__RZ(double %229, ptr %11, ptr null)
  %230 = getelementptr inbounds nuw i8, ptr %12, i64 816
  %231 = load double, ptr %230, align 8, !tbaa !1
  tail call void @__catalyst__qis__RY(double %231, ptr %11, ptr null)
  %232 = getelementptr inbounds nuw i8, ptr %12, i64 824
  %233 = load double, ptr %232, align 8, !tbaa !1
  tail call void @__catalyst__qis__RZ(double %233, ptr %11, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %51, ptr %11, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %11, ptr %51, ptr null)
  %234 = tail call i64 @__catalyst__qis__NamedObs(i64 3, ptr %71)
  %235 = tail call double @__catalyst__qis__Expval(i64 %234)
  %236 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %237 = ptrtoint ptr %236 to i64
  %238 = add i64 %237, 63
  %239 = and i64 %238, -64
  %240 = inttoptr i64 %239 to ptr
  store double %235, ptr %240, align 64, !tbaa !1
  tail call void @__catalyst__rt__qubit_release_array(ptr %9)
  tail call void @__catalyst__rt__device_release()
  %241 = load double, ptr %240, align 64, !tbaa !1
  %242 = extractvalue { ptr, ptr, i64 } %8, 1
  store double %241, ptr %242, align 8, !tbaa !1
  ret void
}

define noalias noundef ptr @qnode_forward_0.quantum.augfwd(ptr %0, ptr readnone captures(none) %1, ptr %2, ptr readnone captures(none) %3, ptr %4, ptr readnone captures(none) %5, ptr %6, ptr readnone captures(none) %7) {
  tail call void @qnode_forward_0.quantum(ptr %0, ptr %2, ptr %4, ptr %6)
  ret ptr null
}

define void @qnode_forward_0.preprocess(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, ptr %9, ptr %10, i64 %11, i64 %12, i64 %13, i64 %14, ptr readnone captures(none) %15, ptr writeonly captures(none) initializes((0, 8)) %16, i64 %17) {
.preheader.preheader:
  %18 = alloca { ptr, ptr, i64 }, align 8
  %19 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, align 8
  %20 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, align 8
  %21 = alloca { ptr, ptr, i64, [3 x i64], [3 x i64] }, align 8
  %.idx = shl i64 %14, 3
  %22 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 %.idx)
  %23 = getelementptr inbounds nuw i8, ptr %1, i64 296
  %24 = load float, ptr %23, align 4, !tbaa !4
  %25 = getelementptr inbounds nuw i8, ptr %1, i64 292
  %26 = load float, ptr %25, align 4, !tbaa !4
  %27 = getelementptr inbounds nuw i8, ptr %1, i64 288
  %28 = load float, ptr %27, align 4, !tbaa !4
  %29 = getelementptr inbounds nuw i8, ptr %1, i64 200
  %30 = load float, ptr %29, align 4, !tbaa !4
  %31 = getelementptr inbounds nuw i8, ptr %1, i64 196
  %32 = load float, ptr %31, align 4, !tbaa !4
  %33 = getelementptr inbounds nuw i8, ptr %1, i64 192
  %34 = load float, ptr %33, align 4, !tbaa !4
  %35 = getelementptr inbounds nuw i8, ptr %1, i64 176
  %36 = load float, ptr %35, align 4, !tbaa !4
  %37 = getelementptr inbounds nuw i8, ptr %1, i64 172
  %38 = load float, ptr %37, align 4, !tbaa !4
  %39 = getelementptr inbounds nuw i8, ptr %1, i64 168
  %40 = load float, ptr %39, align 4, !tbaa !4
  %41 = getelementptr inbounds nuw i8, ptr %1, i64 92
  %42 = load float, ptr %41, align 4, !tbaa !4
  %43 = getelementptr inbounds nuw i8, ptr %1, i64 88
  %44 = load float, ptr %43, align 4, !tbaa !4
  %45 = getelementptr inbounds nuw i8, ptr %1, i64 84
  %46 = load float, ptr %45, align 4, !tbaa !4
  %47 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 96)
  %48 = ptrtoint ptr %47 to i64
  %49 = add i64 %48, 63
  %50 = and i64 %49, -64
  %51 = inttoptr i64 %50 to ptr
  store float 0x400921FB60000000, ptr %51, align 64, !tbaa !4
  %52 = getelementptr inbounds nuw i8, ptr %51, i64 4
  store float 0x400921FB60000000, ptr %52, align 4, !tbaa !4
  %53 = getelementptr inbounds nuw i8, ptr %51, i64 8
  store float 0x400921FB60000000, ptr %53, align 8, !tbaa !4
  %54 = getelementptr inbounds nuw i8, ptr %51, i64 12
  store float 0x400921FB60000000, ptr %54, align 4, !tbaa !4
  %55 = getelementptr inbounds nuw i8, ptr %51, i64 16
  store float 0x400921FB60000000, ptr %55, align 16, !tbaa !4
  %56 = getelementptr inbounds nuw i8, ptr %51, i64 20
  store float 0x400921FB60000000, ptr %56, align 4, !tbaa !4
  %57 = getelementptr inbounds nuw i8, ptr %51, i64 24
  store float 0x400921FB60000000, ptr %57, align 8, !tbaa !4
  %58 = getelementptr inbounds nuw i8, ptr %51, i64 28
  store float 0x400921FB60000000, ptr %58, align 4, !tbaa !4
  %59 = load float, ptr %10, align 4, !tbaa !4
  %60 = fmul float %59, 0x400921FB60000000
  store float %60, ptr %51, align 64, !tbaa !4
  %61 = getelementptr inbounds nuw i8, ptr %10, i64 4
  %62 = load float, ptr %61, align 4, !tbaa !4
  %63 = fmul float %62, 0x400921FB60000000
  store float %63, ptr %52, align 4, !tbaa !4
  %64 = getelementptr inbounds nuw i8, ptr %10, i64 8
  %65 = load float, ptr %64, align 4, !tbaa !4
  %66 = fmul float %65, 0x400921FB60000000
  store float %66, ptr %53, align 8, !tbaa !4
  %67 = getelementptr inbounds nuw i8, ptr %10, i64 12
  %68 = load float, ptr %67, align 4, !tbaa !4
  %69 = fmul float %68, 0x400921FB60000000
  store float %69, ptr %54, align 4, !tbaa !4
  %70 = getelementptr inbounds nuw i8, ptr %10, i64 16
  %71 = load float, ptr %70, align 4, !tbaa !4
  %72 = fmul float %71, 0x400921FB60000000
  store float %72, ptr %55, align 16, !tbaa !4
  %73 = getelementptr inbounds nuw i8, ptr %10, i64 20
  %74 = load float, ptr %73, align 4, !tbaa !4
  %75 = fmul float %74, 0x400921FB60000000
  store float %75, ptr %56, align 4, !tbaa !4
  %76 = getelementptr inbounds nuw i8, ptr %10, i64 24
  %77 = load float, ptr %76, align 4, !tbaa !4
  %78 = fmul float %77, 0x400921FB60000000
  store float %78, ptr %57, align 8, !tbaa !4
  %79 = getelementptr inbounds nuw i8, ptr %10, i64 28
  %80 = load float, ptr %79, align 4, !tbaa !4
  %81 = fmul float %80, 0x400921FB60000000
  store float %81, ptr %58, align 4, !tbaa !4
  %82 = fpext float %81 to double
  store double %82, ptr %22, align 8, !tbaa !1
  %83 = fpext float %46 to double
  %84 = getelementptr inbounds nuw i8, ptr %22, i64 8
  store double %83, ptr %84, align 8, !tbaa !1
  %85 = fpext float %44 to double
  %86 = getelementptr inbounds nuw i8, ptr %22, i64 16
  store double %85, ptr %86, align 8, !tbaa !1
  %87 = fpext float %42 to double
  %88 = getelementptr inbounds nuw i8, ptr %22, i64 24
  store double %87, ptr %88, align 8, !tbaa !1
  %89 = getelementptr inbounds nuw i8, ptr %1, i64 80
  %90 = load float, ptr %89, align 4, !tbaa !4
  %91 = getelementptr inbounds nuw i8, ptr %1, i64 76
  %92 = load float, ptr %91, align 4, !tbaa !4
  %93 = getelementptr inbounds nuw i8, ptr %1, i64 72
  %94 = load float, ptr %93, align 4, !tbaa !4
  %95 = fpext float %78 to double
  %96 = getelementptr inbounds nuw i8, ptr %22, i64 32
  store double %95, ptr %96, align 8, !tbaa !1
  %97 = fpext float %94 to double
  %98 = getelementptr inbounds nuw i8, ptr %22, i64 40
  store double %97, ptr %98, align 8, !tbaa !1
  %99 = fpext float %92 to double
  %100 = getelementptr inbounds nuw i8, ptr %22, i64 48
  store double %99, ptr %100, align 8, !tbaa !1
  %101 = fpext float %90 to double
  %102 = getelementptr inbounds nuw i8, ptr %22, i64 56
  store double %101, ptr %102, align 8, !tbaa !1
  %103 = getelementptr inbounds nuw i8, ptr %1, i64 68
  %104 = load float, ptr %103, align 4, !tbaa !4
  %105 = getelementptr inbounds nuw i8, ptr %1, i64 64
  %106 = load float, ptr %105, align 4, !tbaa !4
  %107 = getelementptr inbounds nuw i8, ptr %1, i64 60
  %108 = load float, ptr %107, align 4, !tbaa !4
  %109 = fpext float %75 to double
  %110 = getelementptr inbounds nuw i8, ptr %22, i64 64
  store double %109, ptr %110, align 8, !tbaa !1
  %111 = fpext float %108 to double
  %112 = getelementptr inbounds nuw i8, ptr %22, i64 72
  store double %111, ptr %112, align 8, !tbaa !1
  %113 = fpext float %106 to double
  %114 = getelementptr inbounds nuw i8, ptr %22, i64 80
  store double %113, ptr %114, align 8, !tbaa !1
  %115 = fpext float %104 to double
  %116 = getelementptr inbounds nuw i8, ptr %22, i64 88
  store double %115, ptr %116, align 8, !tbaa !1
  %117 = getelementptr inbounds nuw i8, ptr %1, i64 56
  %118 = load float, ptr %117, align 4, !tbaa !4
  %119 = getelementptr inbounds nuw i8, ptr %1, i64 52
  %120 = load float, ptr %119, align 4, !tbaa !4
  %121 = getelementptr inbounds nuw i8, ptr %1, i64 48
  %122 = load float, ptr %121, align 4, !tbaa !4
  %123 = fpext float %72 to double
  %124 = getelementptr inbounds nuw i8, ptr %22, i64 96
  store double %123, ptr %124, align 8, !tbaa !1
  %125 = fpext float %122 to double
  %126 = getelementptr inbounds nuw i8, ptr %22, i64 104
  store double %125, ptr %126, align 8, !tbaa !1
  %127 = fpext float %120 to double
  %128 = getelementptr inbounds nuw i8, ptr %22, i64 112
  store double %127, ptr %128, align 8, !tbaa !1
  %129 = fpext float %118 to double
  %130 = getelementptr inbounds nuw i8, ptr %22, i64 120
  store double %129, ptr %130, align 8, !tbaa !1
  %131 = getelementptr inbounds nuw i8, ptr %1, i64 44
  %132 = load float, ptr %131, align 4, !tbaa !4
  %133 = getelementptr inbounds nuw i8, ptr %1, i64 40
  %134 = load float, ptr %133, align 4, !tbaa !4
  %135 = getelementptr inbounds nuw i8, ptr %1, i64 36
  %136 = load float, ptr %135, align 4, !tbaa !4
  %137 = fpext float %69 to double
  %138 = getelementptr inbounds nuw i8, ptr %22, i64 128
  store double %137, ptr %138, align 8, !tbaa !1
  %139 = fpext float %136 to double
  %140 = getelementptr inbounds nuw i8, ptr %22, i64 136
  store double %139, ptr %140, align 8, !tbaa !1
  %141 = fpext float %134 to double
  %142 = getelementptr inbounds nuw i8, ptr %22, i64 144
  store double %141, ptr %142, align 8, !tbaa !1
  %143 = fpext float %132 to double
  %144 = getelementptr inbounds nuw i8, ptr %22, i64 152
  store double %143, ptr %144, align 8, !tbaa !1
  %145 = getelementptr inbounds nuw i8, ptr %1, i64 32
  %146 = load float, ptr %145, align 4, !tbaa !4
  %147 = getelementptr inbounds nuw i8, ptr %1, i64 28
  %148 = load float, ptr %147, align 4, !tbaa !4
  %149 = getelementptr inbounds nuw i8, ptr %1, i64 24
  %150 = load float, ptr %149, align 4, !tbaa !4
  %151 = load float, ptr %53, align 8, !tbaa !4
  %152 = fpext float %151 to double
  %153 = getelementptr inbounds nuw i8, ptr %22, i64 160
  store double %152, ptr %153, align 8, !tbaa !1
  %154 = fpext float %150 to double
  %155 = getelementptr inbounds nuw i8, ptr %22, i64 168
  store double %154, ptr %155, align 8, !tbaa !1
  %156 = fpext float %148 to double
  %157 = getelementptr inbounds nuw i8, ptr %22, i64 176
  store double %156, ptr %157, align 8, !tbaa !1
  %158 = fpext float %146 to double
  %159 = getelementptr inbounds nuw i8, ptr %22, i64 184
  store double %158, ptr %159, align 8, !tbaa !1
  %160 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %161 = load float, ptr %160, align 4, !tbaa !4
  %162 = getelementptr inbounds nuw i8, ptr %1, i64 4
  %163 = load float, ptr %162, align 4, !tbaa !4
  %164 = load float, ptr %1, align 4, !tbaa !4
  %165 = load float, ptr %51, align 64, !tbaa !4
  %166 = fpext float %165 to double
  %167 = getelementptr inbounds nuw i8, ptr %22, i64 192
  store double %166, ptr %167, align 8, !tbaa !1
  %168 = fpext float %164 to double
  %169 = getelementptr inbounds nuw i8, ptr %22, i64 200
  store double %168, ptr %169, align 8, !tbaa !1
  %170 = fpext float %163 to double
  %171 = getelementptr inbounds nuw i8, ptr %22, i64 208
  store double %170, ptr %171, align 8, !tbaa !1
  %172 = fpext float %161 to double
  %173 = getelementptr inbounds nuw i8, ptr %22, i64 216
  store double %172, ptr %173, align 8, !tbaa !1
  %174 = getelementptr inbounds nuw i8, ptr %1, i64 20
  %175 = load float, ptr %174, align 4, !tbaa !4
  %176 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %177 = load float, ptr %176, align 4, !tbaa !4
  %178 = getelementptr inbounds nuw i8, ptr %1, i64 12
  %179 = load float, ptr %178, align 4, !tbaa !4
  %180 = load float, ptr %52, align 4, !tbaa !4
  tail call void @_mlir_memref_to_llvm_free(ptr %47)
  %181 = fpext float %180 to double
  %182 = getelementptr inbounds nuw i8, ptr %22, i64 224
  store double %181, ptr %182, align 8, !tbaa !1
  %183 = fpext float %179 to double
  %184 = getelementptr inbounds nuw i8, ptr %22, i64 232
  store double %183, ptr %184, align 8, !tbaa !1
  %185 = fpext float %177 to double
  %186 = getelementptr inbounds nuw i8, ptr %22, i64 240
  store double %185, ptr %186, align 8, !tbaa !1
  %187 = fpext float %175 to double
  %188 = getelementptr inbounds nuw i8, ptr %22, i64 248
  store double %187, ptr %188, align 8, !tbaa !1
  %189 = fpext float %40 to double
  %190 = getelementptr inbounds nuw i8, ptr %22, i64 256
  store double %189, ptr %190, align 8, !tbaa !1
  %191 = fpext float %38 to double
  %192 = getelementptr inbounds nuw i8, ptr %22, i64 264
  store double %191, ptr %192, align 8, !tbaa !1
  %193 = fpext float %36 to double
  %194 = getelementptr inbounds nuw i8, ptr %22, i64 272
  store double %193, ptr %194, align 8, !tbaa !1
  %195 = getelementptr inbounds nuw i8, ptr %1, i64 152
  %196 = load float, ptr %195, align 4, !tbaa !4
  %197 = getelementptr inbounds nuw i8, ptr %1, i64 148
  %198 = load float, ptr %197, align 4, !tbaa !4
  %199 = getelementptr inbounds nuw i8, ptr %1, i64 144
  %200 = load float, ptr %199, align 4, !tbaa !4
  %201 = fpext float %200 to double
  %202 = getelementptr inbounds nuw i8, ptr %22, i64 280
  store double %201, ptr %202, align 8, !tbaa !1
  %203 = fpext float %198 to double
  %204 = getelementptr inbounds nuw i8, ptr %22, i64 288
  store double %203, ptr %204, align 8, !tbaa !1
  %205 = fpext float %196 to double
  %206 = getelementptr inbounds nuw i8, ptr %22, i64 296
  store double %205, ptr %206, align 8, !tbaa !1
  %207 = getelementptr inbounds nuw i8, ptr %1, i64 104
  %208 = load float, ptr %207, align 4, !tbaa !4
  %209 = getelementptr inbounds nuw i8, ptr %1, i64 100
  %210 = load float, ptr %209, align 4, !tbaa !4
  %211 = getelementptr inbounds nuw i8, ptr %1, i64 96
  %212 = load float, ptr %211, align 4, !tbaa !4
  %213 = fpext float %212 to double
  %214 = getelementptr inbounds nuw i8, ptr %22, i64 304
  store double %213, ptr %214, align 8, !tbaa !1
  %215 = fpext float %210 to double
  %216 = getelementptr inbounds nuw i8, ptr %22, i64 312
  store double %215, ptr %216, align 8, !tbaa !1
  %217 = fpext float %208 to double
  %218 = getelementptr inbounds nuw i8, ptr %22, i64 320
  store double %217, ptr %218, align 8, !tbaa !1
  %219 = getelementptr inbounds nuw i8, ptr %1, i64 128
  %220 = load float, ptr %219, align 4, !tbaa !4
  %221 = getelementptr inbounds nuw i8, ptr %1, i64 124
  %222 = load float, ptr %221, align 4, !tbaa !4
  %223 = getelementptr inbounds nuw i8, ptr %1, i64 120
  %224 = load float, ptr %223, align 4, !tbaa !4
  %225 = fpext float %224 to double
  %226 = getelementptr inbounds nuw i8, ptr %22, i64 328
  store double %225, ptr %226, align 8, !tbaa !1
  %227 = fpext float %222 to double
  %228 = getelementptr inbounds nuw i8, ptr %22, i64 336
  store double %227, ptr %228, align 8, !tbaa !1
  %229 = fpext float %220 to double
  %230 = getelementptr inbounds nuw i8, ptr %22, i64 344
  store double %229, ptr %230, align 8, !tbaa !1
  %231 = fpext float %34 to double
  %232 = getelementptr inbounds nuw i8, ptr %22, i64 352
  store double %231, ptr %232, align 8, !tbaa !1
  %233 = fpext float %32 to double
  %234 = getelementptr inbounds nuw i8, ptr %22, i64 360
  store double %233, ptr %234, align 8, !tbaa !1
  %235 = fpext float %30 to double
  %236 = getelementptr inbounds nuw i8, ptr %22, i64 368
  store double %235, ptr %236, align 8, !tbaa !1
  %237 = getelementptr inbounds nuw i8, ptr %1, i64 236
  %238 = load float, ptr %237, align 4, !tbaa !4
  %239 = getelementptr inbounds nuw i8, ptr %1, i64 232
  %240 = load float, ptr %239, align 4, !tbaa !4
  %241 = getelementptr inbounds nuw i8, ptr %1, i64 228
  %242 = load float, ptr %241, align 4, !tbaa !4
  %243 = getelementptr inbounds nuw i8, ptr %1, i64 164
  %244 = load float, ptr %243, align 4, !tbaa !4
  %245 = getelementptr inbounds nuw i8, ptr %1, i64 160
  %246 = load float, ptr %245, align 4, !tbaa !4
  %247 = getelementptr inbounds nuw i8, ptr %1, i64 156
  %248 = load float, ptr %247, align 4, !tbaa !4
  %249 = fpext float %248 to double
  %250 = getelementptr inbounds nuw i8, ptr %22, i64 376
  store double %249, ptr %250, align 8, !tbaa !1
  %251 = fpext float %246 to double
  %252 = getelementptr inbounds nuw i8, ptr %22, i64 384
  store double %251, ptr %252, align 8, !tbaa !1
  %253 = fpext float %244 to double
  %254 = getelementptr inbounds nuw i8, ptr %22, i64 392
  store double %253, ptr %254, align 8, !tbaa !1
  %255 = getelementptr inbounds nuw i8, ptr %1, i64 116
  %256 = load float, ptr %255, align 4, !tbaa !4
  %257 = getelementptr inbounds nuw i8, ptr %1, i64 112
  %258 = load float, ptr %257, align 4, !tbaa !4
  %259 = getelementptr inbounds nuw i8, ptr %1, i64 108
  %260 = load float, ptr %259, align 4, !tbaa !4
  %261 = fpext float %260 to double
  %262 = getelementptr inbounds nuw i8, ptr %22, i64 400
  store double %261, ptr %262, align 8, !tbaa !1
  %263 = fpext float %258 to double
  %264 = getelementptr inbounds nuw i8, ptr %22, i64 408
  store double %263, ptr %264, align 8, !tbaa !1
  %265 = fpext float %256 to double
  %266 = getelementptr inbounds nuw i8, ptr %22, i64 416
  store double %265, ptr %266, align 8, !tbaa !1
  %267 = getelementptr inbounds nuw i8, ptr %1, i64 140
  %268 = load float, ptr %267, align 4, !tbaa !4
  %269 = getelementptr inbounds nuw i8, ptr %1, i64 136
  %270 = load float, ptr %269, align 4, !tbaa !4
  %271 = getelementptr inbounds nuw i8, ptr %1, i64 132
  %272 = load float, ptr %271, align 4, !tbaa !4
  %273 = fpext float %272 to double
  %274 = getelementptr inbounds nuw i8, ptr %22, i64 424
  store double %273, ptr %274, align 8, !tbaa !1
  %275 = fpext float %270 to double
  %276 = getelementptr inbounds nuw i8, ptr %22, i64 432
  store double %275, ptr %276, align 8, !tbaa !1
  %277 = fpext float %268 to double
  %278 = getelementptr inbounds nuw i8, ptr %22, i64 440
  store double %277, ptr %278, align 8, !tbaa !1
  %279 = fpext float %242 to double
  %280 = getelementptr inbounds nuw i8, ptr %22, i64 448
  store double %279, ptr %280, align 8, !tbaa !1
  %281 = fpext float %240 to double
  %282 = getelementptr inbounds nuw i8, ptr %22, i64 456
  store double %281, ptr %282, align 8, !tbaa !1
  %283 = fpext float %238 to double
  %284 = getelementptr inbounds nuw i8, ptr %22, i64 464
  store double %283, ptr %284, align 8, !tbaa !1
  %285 = getelementptr inbounds nuw i8, ptr %1, i64 224
  %286 = load float, ptr %285, align 4, !tbaa !4
  %287 = getelementptr inbounds nuw i8, ptr %1, i64 220
  %288 = load float, ptr %287, align 4, !tbaa !4
  %289 = getelementptr inbounds nuw i8, ptr %1, i64 216
  %290 = load float, ptr %289, align 4, !tbaa !4
  %291 = fpext float %290 to double
  %292 = getelementptr inbounds nuw i8, ptr %22, i64 472
  store double %291, ptr %292, align 8, !tbaa !1
  %293 = fpext float %288 to double
  %294 = getelementptr inbounds nuw i8, ptr %22, i64 480
  store double %293, ptr %294, align 8, !tbaa !1
  %295 = fpext float %286 to double
  %296 = getelementptr inbounds nuw i8, ptr %22, i64 488
  store double %295, ptr %296, align 8, !tbaa !1
  %297 = getelementptr inbounds nuw i8, ptr %1, i64 260
  %298 = load float, ptr %297, align 4, !tbaa !4
  %299 = getelementptr inbounds nuw i8, ptr %1, i64 256
  %300 = load float, ptr %299, align 4, !tbaa !4
  %301 = getelementptr inbounds nuw i8, ptr %1, i64 252
  %302 = load float, ptr %301, align 4, !tbaa !4
  %303 = getelementptr inbounds nuw i8, ptr %1, i64 188
  %304 = load float, ptr %303, align 4, !tbaa !4
  %305 = getelementptr inbounds nuw i8, ptr %1, i64 184
  %306 = load float, ptr %305, align 4, !tbaa !4
  %307 = getelementptr inbounds nuw i8, ptr %1, i64 180
  %308 = load float, ptr %307, align 4, !tbaa !4
  %309 = fpext float %308 to double
  %310 = getelementptr inbounds nuw i8, ptr %22, i64 496
  store double %309, ptr %310, align 8, !tbaa !1
  %311 = fpext float %306 to double
  %312 = getelementptr inbounds nuw i8, ptr %22, i64 504
  store double %311, ptr %312, align 8, !tbaa !1
  %313 = fpext float %304 to double
  %314 = getelementptr inbounds nuw i8, ptr %22, i64 512
  store double %313, ptr %314, align 8, !tbaa !1
  %315 = fpext float %302 to double
  %316 = getelementptr inbounds nuw i8, ptr %22, i64 520
  store double %315, ptr %316, align 8, !tbaa !1
  %317 = fpext float %300 to double
  %318 = getelementptr inbounds nuw i8, ptr %22, i64 528
  store double %317, ptr %318, align 8, !tbaa !1
  %319 = fpext float %298 to double
  %320 = getelementptr inbounds nuw i8, ptr %22, i64 536
  store double %319, ptr %320, align 8, !tbaa !1
  %321 = fpext float %28 to double
  %322 = getelementptr inbounds nuw i8, ptr %22, i64 544
  store double %321, ptr %322, align 8, !tbaa !1
  %323 = fpext float %26 to double
  %324 = getelementptr inbounds nuw i8, ptr %22, i64 552
  store double %323, ptr %324, align 8, !tbaa !1
  %325 = fpext float %24 to double
  %326 = getelementptr inbounds nuw i8, ptr %22, i64 560
  store double %325, ptr %326, align 8, !tbaa !1
  %327 = getelementptr inbounds nuw i8, ptr %1, i64 344
  %328 = load float, ptr %327, align 4, !tbaa !4
  %329 = getelementptr inbounds nuw i8, ptr %1, i64 340
  %330 = load float, ptr %329, align 4, !tbaa !4
  %331 = getelementptr inbounds nuw i8, ptr %1, i64 336
  %332 = load float, ptr %331, align 4, !tbaa !4
  %333 = getelementptr inbounds nuw i8, ptr %1, i64 284
  %334 = load float, ptr %333, align 4, !tbaa !4
  %335 = getelementptr inbounds nuw i8, ptr %1, i64 280
  %336 = load float, ptr %335, align 4, !tbaa !4
  %337 = getelementptr inbounds nuw i8, ptr %1, i64 276
  %338 = load float, ptr %337, align 4, !tbaa !4
  %339 = fpext float %338 to double
  %340 = getelementptr inbounds nuw i8, ptr %22, i64 568
  store double %339, ptr %340, align 8, !tbaa !1
  %341 = fpext float %336 to double
  %342 = getelementptr inbounds nuw i8, ptr %22, i64 576
  store double %341, ptr %342, align 8, !tbaa !1
  %343 = fpext float %334 to double
  %344 = getelementptr inbounds nuw i8, ptr %22, i64 584
  store double %343, ptr %344, align 8, !tbaa !1
  %345 = getelementptr inbounds nuw i8, ptr %1, i64 212
  %346 = load float, ptr %345, align 4, !tbaa !4
  %347 = getelementptr inbounds nuw i8, ptr %1, i64 208
  %348 = load float, ptr %347, align 4, !tbaa !4
  %349 = getelementptr inbounds nuw i8, ptr %1, i64 204
  %350 = load float, ptr %349, align 4, !tbaa !4
  %351 = fpext float %350 to double
  %352 = getelementptr inbounds nuw i8, ptr %22, i64 592
  store double %351, ptr %352, align 8, !tbaa !1
  %353 = fpext float %348 to double
  %354 = getelementptr inbounds nuw i8, ptr %22, i64 600
  store double %353, ptr %354, align 8, !tbaa !1
  %355 = fpext float %346 to double
  %356 = getelementptr inbounds nuw i8, ptr %22, i64 608
  store double %355, ptr %356, align 8, !tbaa !1
  %357 = getelementptr inbounds nuw i8, ptr %1, i64 248
  %358 = load float, ptr %357, align 4, !tbaa !4
  %359 = getelementptr inbounds nuw i8, ptr %1, i64 244
  %360 = load float, ptr %359, align 4, !tbaa !4
  %361 = getelementptr inbounds nuw i8, ptr %1, i64 240
  %362 = load float, ptr %361, align 4, !tbaa !4
  %363 = fpext float %362 to double
  %364 = getelementptr inbounds nuw i8, ptr %22, i64 616
  store double %363, ptr %364, align 8, !tbaa !1
  %365 = fpext float %360 to double
  %366 = getelementptr inbounds nuw i8, ptr %22, i64 624
  store double %365, ptr %366, align 8, !tbaa !1
  %367 = fpext float %358 to double
  %368 = getelementptr inbounds nuw i8, ptr %22, i64 632
  store double %367, ptr %368, align 8, !tbaa !1
  %369 = fpext float %332 to double
  %370 = getelementptr inbounds nuw i8, ptr %22, i64 640
  store double %369, ptr %370, align 8, !tbaa !1
  %371 = fpext float %330 to double
  %372 = getelementptr inbounds nuw i8, ptr %22, i64 648
  store double %371, ptr %372, align 8, !tbaa !1
  %373 = fpext float %328 to double
  %374 = getelementptr inbounds nuw i8, ptr %22, i64 656
  store double %373, ptr %374, align 8, !tbaa !1
  %375 = getelementptr inbounds nuw i8, ptr %1, i64 308
  %376 = load float, ptr %375, align 4, !tbaa !4
  %377 = getelementptr inbounds nuw i8, ptr %1, i64 304
  %378 = load float, ptr %377, align 4, !tbaa !4
  %379 = getelementptr inbounds nuw i8, ptr %1, i64 300
  %380 = load float, ptr %379, align 4, !tbaa !4
  %381 = getelementptr inbounds nuw i8, ptr %1, i64 272
  %382 = load float, ptr %381, align 4, !tbaa !4
  %383 = getelementptr inbounds nuw i8, ptr %1, i64 268
  %384 = load float, ptr %383, align 4, !tbaa !4
  %385 = getelementptr inbounds nuw i8, ptr %1, i64 264
  %386 = load float, ptr %385, align 4, !tbaa !4
  %387 = fpext float %386 to double
  %388 = getelementptr inbounds nuw i8, ptr %22, i64 664
  store double %387, ptr %388, align 8, !tbaa !1
  %389 = fpext float %384 to double
  %390 = getelementptr inbounds nuw i8, ptr %22, i64 672
  store double %389, ptr %390, align 8, !tbaa !1
  %391 = fpext float %382 to double
  %392 = getelementptr inbounds nuw i8, ptr %22, i64 680
  store double %391, ptr %392, align 8, !tbaa !1
  %393 = fpext float %380 to double
  %394 = getelementptr inbounds nuw i8, ptr %22, i64 688
  store double %393, ptr %394, align 8, !tbaa !1
  %395 = fpext float %378 to double
  %396 = getelementptr inbounds nuw i8, ptr %22, i64 696
  store double %395, ptr %396, align 8, !tbaa !1
  %397 = fpext float %376 to double
  %398 = getelementptr inbounds nuw i8, ptr %22, i64 704
  store double %397, ptr %398, align 8, !tbaa !1
  %399 = getelementptr inbounds nuw i8, ptr %1, i64 356
  %400 = load float, ptr %399, align 4, !tbaa !4
  %401 = getelementptr inbounds nuw i8, ptr %1, i64 352
  %402 = load float, ptr %401, align 4, !tbaa !4
  %403 = getelementptr inbounds nuw i8, ptr %1, i64 348
  %404 = load float, ptr %403, align 4, !tbaa !4
  %405 = fpext float %404 to double
  %406 = getelementptr inbounds nuw i8, ptr %22, i64 712
  store double %405, ptr %406, align 8, !tbaa !1
  %407 = fpext float %402 to double
  %408 = getelementptr inbounds nuw i8, ptr %22, i64 720
  store double %407, ptr %408, align 8, !tbaa !1
  %409 = fpext float %400 to double
  %410 = getelementptr inbounds nuw i8, ptr %22, i64 728
  store double %409, ptr %410, align 8, !tbaa !1
  %411 = getelementptr inbounds nuw i8, ptr %1, i64 320
  %412 = load float, ptr %411, align 4, !tbaa !4
  %413 = getelementptr inbounds nuw i8, ptr %1, i64 316
  %414 = load float, ptr %413, align 4, !tbaa !4
  %415 = getelementptr inbounds nuw i8, ptr %1, i64 312
  %416 = load float, ptr %415, align 4, !tbaa !4
  %417 = fpext float %416 to double
  %418 = getelementptr inbounds nuw i8, ptr %22, i64 736
  store double %417, ptr %418, align 8, !tbaa !1
  %419 = fpext float %414 to double
  %420 = getelementptr inbounds nuw i8, ptr %22, i64 744
  store double %419, ptr %420, align 8, !tbaa !1
  %421 = fpext float %412 to double
  %422 = getelementptr inbounds nuw i8, ptr %22, i64 752
  store double %421, ptr %422, align 8, !tbaa !1
  %423 = getelementptr inbounds nuw i8, ptr %1, i64 368
  %424 = load float, ptr %423, align 4, !tbaa !4
  %425 = getelementptr inbounds nuw i8, ptr %1, i64 364
  %426 = load float, ptr %425, align 4, !tbaa !4
  %427 = getelementptr inbounds nuw i8, ptr %1, i64 360
  %428 = load float, ptr %427, align 4, !tbaa !4
  %429 = fpext float %428 to double
  %430 = getelementptr inbounds nuw i8, ptr %22, i64 760
  store double %429, ptr %430, align 8, !tbaa !1
  %431 = fpext float %426 to double
  %432 = getelementptr inbounds nuw i8, ptr %22, i64 768
  store double %431, ptr %432, align 8, !tbaa !1
  %433 = fpext float %424 to double
  %434 = getelementptr inbounds nuw i8, ptr %22, i64 776
  store double %433, ptr %434, align 8, !tbaa !1
  %435 = getelementptr inbounds nuw i8, ptr %1, i64 332
  %436 = load float, ptr %435, align 4, !tbaa !4
  %437 = getelementptr inbounds nuw i8, ptr %1, i64 328
  %438 = load float, ptr %437, align 4, !tbaa !4
  %439 = getelementptr inbounds nuw i8, ptr %1, i64 324
  %440 = load float, ptr %439, align 4, !tbaa !4
  %441 = fpext float %440 to double
  %442 = getelementptr inbounds nuw i8, ptr %22, i64 784
  store double %441, ptr %442, align 8, !tbaa !1
  %443 = fpext float %438 to double
  %444 = getelementptr inbounds nuw i8, ptr %22, i64 792
  store double %443, ptr %444, align 8, !tbaa !1
  %445 = fpext float %436 to double
  %446 = getelementptr inbounds nuw i8, ptr %22, i64 800
  store double %445, ptr %446, align 8, !tbaa !1
  %447 = getelementptr inbounds nuw i8, ptr %1, i64 380
  %448 = load float, ptr %447, align 4, !tbaa !4
  %449 = getelementptr inbounds nuw i8, ptr %1, i64 376
  %450 = load float, ptr %449, align 4, !tbaa !4
  %451 = getelementptr inbounds nuw i8, ptr %1, i64 372
  %452 = load float, ptr %451, align 4, !tbaa !4
  %453 = fpext float %452 to double
  %454 = getelementptr inbounds nuw i8, ptr %22, i64 808
  store double %453, ptr %454, align 8, !tbaa !1
  %455 = fpext float %450 to double
  %456 = getelementptr inbounds nuw i8, ptr %22, i64 816
  store double %455, ptr %456, align 8, !tbaa !1
  %457 = fpext float %448 to double
  %458 = getelementptr inbounds nuw i8, ptr %22, i64 824
  store double %457, ptr %458, align 8, !tbaa !1
  store ptr %0, ptr %21, align 8
  %.fca.1.gep = getelementptr inbounds nuw i8, ptr %21, i64 8
  store ptr %1, ptr %.fca.1.gep, align 8
  %.fca.2.gep = getelementptr inbounds nuw i8, ptr %21, i64 16
  store i64 %2, ptr %.fca.2.gep, align 8
  %.fca.3.0.gep = getelementptr inbounds nuw i8, ptr %21, i64 24
  store i64 %3, ptr %.fca.3.0.gep, align 8
  %.fca.3.1.gep = getelementptr inbounds nuw i8, ptr %21, i64 32
  store i64 %4, ptr %.fca.3.1.gep, align 8
  %.fca.3.2.gep = getelementptr inbounds nuw i8, ptr %21, i64 40
  store i64 %5, ptr %.fca.3.2.gep, align 8
  %.fca.4.0.gep = getelementptr inbounds nuw i8, ptr %21, i64 48
  store i64 %6, ptr %.fca.4.0.gep, align 8
  %.fca.4.1.gep = getelementptr inbounds nuw i8, ptr %21, i64 56
  store i64 %7, ptr %.fca.4.1.gep, align 8
  %.fca.4.2.gep = getelementptr inbounds nuw i8, ptr %21, i64 64
  store i64 %8, ptr %.fca.4.2.gep, align 8
  store ptr %9, ptr %20, align 8
  %.fca.1.gep107 = getelementptr inbounds nuw i8, ptr %20, i64 8
  store ptr %10, ptr %.fca.1.gep107, align 8
  %.fca.2.gep109 = getelementptr inbounds nuw i8, ptr %20, i64 16
  store i64 %11, ptr %.fca.2.gep109, align 8
  %.fca.3.0.gep111 = getelementptr inbounds nuw i8, ptr %20, i64 24
  store i64 %12, ptr %.fca.3.0.gep111, align 8
  %.fca.4.0.gep113 = getelementptr inbounds nuw i8, ptr %20, i64 32
  store i64 %13, ptr %.fca.4.0.gep113, align 8
  store ptr %22, ptr %19, align 8
  %.fca.1.gep117 = getelementptr inbounds nuw i8, ptr %19, i64 8
  store ptr %22, ptr %.fca.1.gep117, align 8
  %.fca.2.gep119 = getelementptr inbounds nuw i8, ptr %19, i64 16
  store i64 0, ptr %.fca.2.gep119, align 8
  %.fca.3.0.gep121 = getelementptr inbounds nuw i8, ptr %19, i64 24
  store i64 %14, ptr %.fca.3.0.gep121, align 8
  %.fca.4.0.gep123 = getelementptr inbounds nuw i8, ptr %19, i64 32
  store i64 1, ptr %.fca.4.0.gep123, align 8
  %459 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 8)
  store ptr %459, ptr %18, align 8
  %.fca.1.gep127 = getelementptr inbounds nuw i8, ptr %18, i64 8
  store ptr %459, ptr %.fca.1.gep127, align 8
  %.fca.2.gep129 = getelementptr inbounds nuw i8, ptr %18, i64 16
  store i64 0, ptr %.fca.2.gep129, align 8
  call void @qnode_forward_0.quantum(ptr nonnull %21, ptr nonnull %20, ptr nonnull %19, ptr nonnull %18)
  %460 = load double, ptr %459, align 8, !tbaa !1
  store double %460, ptr %16, align 8, !tbaa !1
  ret void
}

define void @setup() local_unnamed_addr {
  tail call void @__catalyst__rt__initialize(ptr null)
  ret void
}

define void @teardown() local_unnamed_addr {
  tail call void @__catalyst__rt__finalize()
  ret void
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #2

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #3

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #4

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #4

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) }
attributes #1 = { noinline }
attributes #2 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #3 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #4 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{!2, !2, i64 0}
!2 = !{!"double", !3, i64 0}
!3 = !{!"Catalyst TBAA"}
!4 = !{!5, !5, i64 0}
!5 = !{!"float", !3, i64 0}
