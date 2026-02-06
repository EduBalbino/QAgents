; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@"{}" = internal constant [3 x i8] c"{}\00"
@LightningGPUSimulator = internal constant [22 x i8] c"LightningGPUSimulator\00"
@"/home/pichau/QAgents/.venv/lib/python3.12/site-packages/pennylane_lightning/liblightning_gpu_catalyst.so" = internal constant [105 x i8] c"/home/pichau/QAgents/.venv/lib/python3.12/site-packages/pennylane_lightning/liblightning_gpu_catalyst.so\00"

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

declare void @_mlir_memref_to_llvm_free(ptr) local_unnamed_addr #0

declare !enzyme_deallocator_fn !1 ptr @_mlir_memref_to_llvm_alloc(i64) local_unnamed_addr #1

define { ptr, ptr, i64, [3 x i64], [3 x i64] } @jit_deriv_qnode_forward(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, ptr %9, ptr %10, i64 %11, i64 %12, i64 %13) local_unnamed_addr {
  %"'ipa5.i" = alloca { ptr, ptr, i64 }, align 8
  %15 = alloca { ptr, ptr, i64 }, align 8
  %"'ipa4.i" = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, align 8
  %16 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, align 8
  %17 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, align 8
  %18 = alloca { ptr, ptr, i64, [3 x i64], [3 x i64] }, align 8
  %19 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %20 = ptrtoint ptr %19 to i64
  %21 = add i64 %20, 63
  %22 = and i64 %21, -64
  %23 = inttoptr i64 %22 to ptr
  store double 1.000000e+00, ptr %23, align 64, !tbaa !2
  %24 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 384)
  %25 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 8)
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 1 dereferenceable(384) %24, i8 0, i64 384, i1 false)
  call void @llvm.lifetime.start.p0(ptr nonnull %"'ipa5.i")
  call void @llvm.lifetime.start.p0(ptr nonnull %15)
  call void @llvm.lifetime.start.p0(ptr nonnull %"'ipa4.i")
  call void @llvm.lifetime.start.p0(ptr nonnull %16)
  call void @llvm.lifetime.start.p0(ptr nonnull %17)
  call void @llvm.lifetime.start.p0(ptr nonnull %18)
  %.fca.1.gep415.i = getelementptr inbounds nuw i8, ptr %"'ipa5.i", i64 8
  %.fca.2.gep416.i = getelementptr inbounds nuw i8, ptr %"'ipa5.i", i64 16
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(24) %"'ipa5.i", i8 0, i64 24, i1 false)
  %.fca.1.gep410.i = getelementptr inbounds nuw i8, ptr %"'ipa4.i", i64 8
  %.fca.2.gep411.i = getelementptr inbounds nuw i8, ptr %"'ipa4.i", i64 16
  %.fca.3.0.gep412.i = getelementptr inbounds nuw i8, ptr %"'ipa4.i", i64 24
  %.fca.4.0.gep413.i = getelementptr inbounds nuw i8, ptr %"'ipa4.i", i64 32
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(40) %"'ipa4.i", i8 0, i64 40, i1 false)
  %"'mi.i" = tail call noalias nonnull ptr @_mlir_memref_to_llvm_alloc(i64 832) #8
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 1 dereferenceable(832) %"'mi.i", i8 0, i64 832, i1 false)
  %26 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 832) #8
  %"'ipg395.i" = getelementptr inbounds nuw i8, ptr %24, i64 296
  %27 = getelementptr inbounds nuw i8, ptr %1, i64 296
  %28 = load float, ptr %27, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %"'ipg394.i" = getelementptr inbounds nuw i8, ptr %24, i64 292
  %29 = getelementptr inbounds nuw i8, ptr %1, i64 292
  %30 = load float, ptr %29, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %"'ipg393.i" = getelementptr inbounds nuw i8, ptr %24, i64 288
  %31 = getelementptr inbounds nuw i8, ptr %1, i64 288
  %32 = load float, ptr %31, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %"'ipg392.i" = getelementptr inbounds nuw i8, ptr %24, i64 200
  %33 = getelementptr inbounds nuw i8, ptr %1, i64 200
  %34 = load float, ptr %33, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %"'ipg391.i" = getelementptr inbounds nuw i8, ptr %24, i64 196
  %35 = getelementptr inbounds nuw i8, ptr %1, i64 196
  %36 = load float, ptr %35, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %"'ipg390.i" = getelementptr inbounds nuw i8, ptr %24, i64 192
  %37 = getelementptr inbounds nuw i8, ptr %1, i64 192
  %38 = load float, ptr %37, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %"'ipg389.i" = getelementptr inbounds nuw i8, ptr %24, i64 176
  %39 = getelementptr inbounds nuw i8, ptr %1, i64 176
  %40 = load float, ptr %39, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %"'ipg388.i" = getelementptr inbounds nuw i8, ptr %24, i64 172
  %41 = getelementptr inbounds nuw i8, ptr %1, i64 172
  %42 = load float, ptr %41, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %"'ipg387.i" = getelementptr inbounds nuw i8, ptr %24, i64 168
  %43 = getelementptr inbounds nuw i8, ptr %1, i64 168
  %44 = load float, ptr %43, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %"'ipg386.i" = getelementptr inbounds nuw i8, ptr %24, i64 92
  %45 = getelementptr inbounds nuw i8, ptr %1, i64 92
  %46 = load float, ptr %45, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %"'ipg385.i" = getelementptr inbounds nuw i8, ptr %24, i64 88
  %47 = getelementptr inbounds nuw i8, ptr %1, i64 88
  %48 = load float, ptr %47, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %"'ipg384.i" = getelementptr inbounds nuw i8, ptr %24, i64 84
  %49 = getelementptr inbounds nuw i8, ptr %1, i64 84
  %50 = load float, ptr %49, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %51 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 96) #8
  %52 = ptrtoint ptr %51 to i64
  %53 = add i64 %52, 63
  %54 = and i64 %53, -64
  %55 = inttoptr i64 %54 to ptr
  store float 0x400921FB60000000, ptr %55, align 64, !tbaa !5
  %56 = getelementptr inbounds nuw i8, ptr %55, i64 4
  store float 0x400921FB60000000, ptr %56, align 4, !tbaa !5
  %57 = getelementptr inbounds nuw i8, ptr %55, i64 8
  store float 0x400921FB60000000, ptr %57, align 8, !tbaa !5
  %58 = getelementptr inbounds nuw i8, ptr %55, i64 12
  store float 0x400921FB60000000, ptr %58, align 4, !tbaa !5
  %59 = getelementptr inbounds nuw i8, ptr %55, i64 16
  store float 0x400921FB60000000, ptr %59, align 16, !tbaa !5
  %60 = getelementptr inbounds nuw i8, ptr %55, i64 20
  store float 0x400921FB60000000, ptr %60, align 4, !tbaa !5
  %61 = getelementptr inbounds nuw i8, ptr %55, i64 24
  store float 0x400921FB60000000, ptr %61, align 8, !tbaa !5
  %62 = getelementptr inbounds nuw i8, ptr %55, i64 28
  store float 0x400921FB60000000, ptr %62, align 4, !tbaa !5
  %63 = load float, ptr %10, align 4, !tbaa !5, !alias.scope !12, !noalias !15
  %64 = fmul float %63, 0x400921FB60000000
  store float %64, ptr %55, align 64, !tbaa !5
  %65 = getelementptr inbounds nuw i8, ptr %10, i64 4
  %66 = load float, ptr %65, align 4, !tbaa !5, !alias.scope !12, !noalias !15
  %67 = fmul float %66, 0x400921FB60000000
  store float %67, ptr %56, align 4, !tbaa !5
  %68 = getelementptr inbounds nuw i8, ptr %10, i64 8
  %69 = load float, ptr %68, align 4, !tbaa !5, !alias.scope !12, !noalias !15
  %70 = fmul float %69, 0x400921FB60000000
  store float %70, ptr %57, align 8, !tbaa !5
  %71 = getelementptr inbounds nuw i8, ptr %10, i64 12
  %72 = load float, ptr %71, align 4, !tbaa !5, !alias.scope !12, !noalias !15
  %73 = fmul float %72, 0x400921FB60000000
  store float %73, ptr %58, align 4, !tbaa !5
  %74 = getelementptr inbounds nuw i8, ptr %10, i64 16
  %75 = load float, ptr %74, align 4, !tbaa !5, !alias.scope !12, !noalias !15
  %76 = fmul float %75, 0x400921FB60000000
  store float %76, ptr %59, align 16, !tbaa !5
  %77 = getelementptr inbounds nuw i8, ptr %10, i64 20
  %78 = load float, ptr %77, align 4, !tbaa !5, !alias.scope !12, !noalias !15
  %79 = fmul float %78, 0x400921FB60000000
  store float %79, ptr %60, align 4, !tbaa !5
  %80 = getelementptr inbounds nuw i8, ptr %10, i64 24
  %81 = load float, ptr %80, align 4, !tbaa !5, !alias.scope !12, !noalias !15
  %82 = fmul float %81, 0x400921FB60000000
  store float %82, ptr %61, align 8, !tbaa !5
  %83 = getelementptr inbounds nuw i8, ptr %10, i64 28
  %84 = load float, ptr %83, align 4, !tbaa !5, !alias.scope !12, !noalias !15
  %85 = fmul float %84, 0x400921FB60000000
  store float %85, ptr %62, align 4, !tbaa !5
  %86 = fpext float %85 to double
  store double %86, ptr %26, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %87 = fpext float %50 to double
  %"'ipg381.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 8
  %88 = getelementptr inbounds nuw i8, ptr %26, i64 8
  store double %87, ptr %88, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %89 = fpext float %48 to double
  %"'ipg378.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 16
  %90 = getelementptr inbounds nuw i8, ptr %26, i64 16
  store double %89, ptr %90, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %91 = fpext float %46 to double
  %"'ipg375.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 24
  %92 = getelementptr inbounds nuw i8, ptr %26, i64 24
  store double %91, ptr %92, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %"'ipg374.i" = getelementptr inbounds nuw i8, ptr %24, i64 80
  %93 = getelementptr inbounds nuw i8, ptr %1, i64 80
  %94 = load float, ptr %93, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %"'ipg373.i" = getelementptr inbounds nuw i8, ptr %24, i64 76
  %95 = getelementptr inbounds nuw i8, ptr %1, i64 76
  %96 = load float, ptr %95, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %"'ipg372.i" = getelementptr inbounds nuw i8, ptr %24, i64 72
  %97 = getelementptr inbounds nuw i8, ptr %1, i64 72
  %98 = load float, ptr %97, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %99 = fpext float %82 to double
  %"'ipg371.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 32
  %100 = getelementptr inbounds nuw i8, ptr %26, i64 32
  store double %99, ptr %100, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %101 = fpext float %98 to double
  %"'ipg368.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 40
  %102 = getelementptr inbounds nuw i8, ptr %26, i64 40
  store double %101, ptr %102, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %103 = fpext float %96 to double
  %"'ipg365.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 48
  %104 = getelementptr inbounds nuw i8, ptr %26, i64 48
  store double %103, ptr %104, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %105 = fpext float %94 to double
  %"'ipg362.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 56
  %106 = getelementptr inbounds nuw i8, ptr %26, i64 56
  store double %105, ptr %106, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %"'ipg361.i" = getelementptr inbounds nuw i8, ptr %24, i64 68
  %107 = getelementptr inbounds nuw i8, ptr %1, i64 68
  %108 = load float, ptr %107, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %"'ipg360.i" = getelementptr inbounds nuw i8, ptr %24, i64 64
  %109 = getelementptr inbounds nuw i8, ptr %1, i64 64
  %110 = load float, ptr %109, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %"'ipg359.i" = getelementptr inbounds nuw i8, ptr %24, i64 60
  %111 = getelementptr inbounds nuw i8, ptr %1, i64 60
  %112 = load float, ptr %111, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %113 = fpext float %79 to double
  %"'ipg358.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 64
  %114 = getelementptr inbounds nuw i8, ptr %26, i64 64
  store double %113, ptr %114, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %115 = fpext float %112 to double
  %"'ipg355.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 72
  %116 = getelementptr inbounds nuw i8, ptr %26, i64 72
  store double %115, ptr %116, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %117 = fpext float %110 to double
  %"'ipg352.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 80
  %118 = getelementptr inbounds nuw i8, ptr %26, i64 80
  store double %117, ptr %118, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %119 = fpext float %108 to double
  %"'ipg349.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 88
  %120 = getelementptr inbounds nuw i8, ptr %26, i64 88
  store double %119, ptr %120, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %"'ipg348.i" = getelementptr inbounds nuw i8, ptr %24, i64 56
  %121 = getelementptr inbounds nuw i8, ptr %1, i64 56
  %122 = load float, ptr %121, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %"'ipg347.i" = getelementptr inbounds nuw i8, ptr %24, i64 52
  %123 = getelementptr inbounds nuw i8, ptr %1, i64 52
  %124 = load float, ptr %123, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %"'ipg346.i" = getelementptr inbounds nuw i8, ptr %24, i64 48
  %125 = getelementptr inbounds nuw i8, ptr %1, i64 48
  %126 = load float, ptr %125, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %127 = fpext float %76 to double
  %"'ipg345.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 96
  %128 = getelementptr inbounds nuw i8, ptr %26, i64 96
  store double %127, ptr %128, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %129 = fpext float %126 to double
  %"'ipg342.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 104
  %130 = getelementptr inbounds nuw i8, ptr %26, i64 104
  store double %129, ptr %130, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %131 = fpext float %124 to double
  %"'ipg339.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 112
  %132 = getelementptr inbounds nuw i8, ptr %26, i64 112
  store double %131, ptr %132, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %133 = fpext float %122 to double
  %"'ipg336.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 120
  %134 = getelementptr inbounds nuw i8, ptr %26, i64 120
  store double %133, ptr %134, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %"'ipg335.i" = getelementptr inbounds nuw i8, ptr %24, i64 44
  %135 = getelementptr inbounds nuw i8, ptr %1, i64 44
  %136 = load float, ptr %135, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %"'ipg334.i" = getelementptr inbounds nuw i8, ptr %24, i64 40
  %137 = getelementptr inbounds nuw i8, ptr %1, i64 40
  %138 = load float, ptr %137, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %"'ipg333.i" = getelementptr inbounds nuw i8, ptr %24, i64 36
  %139 = getelementptr inbounds nuw i8, ptr %1, i64 36
  %140 = load float, ptr %139, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %141 = fpext float %73 to double
  %"'ipg332.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 128
  %142 = getelementptr inbounds nuw i8, ptr %26, i64 128
  store double %141, ptr %142, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %143 = fpext float %140 to double
  %"'ipg329.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 136
  %144 = getelementptr inbounds nuw i8, ptr %26, i64 136
  store double %143, ptr %144, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %145 = fpext float %138 to double
  %"'ipg326.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 144
  %146 = getelementptr inbounds nuw i8, ptr %26, i64 144
  store double %145, ptr %146, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %147 = fpext float %136 to double
  %"'ipg323.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 152
  %148 = getelementptr inbounds nuw i8, ptr %26, i64 152
  store double %147, ptr %148, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %"'ipg322.i" = getelementptr inbounds nuw i8, ptr %24, i64 32
  %149 = getelementptr inbounds nuw i8, ptr %1, i64 32
  %150 = load float, ptr %149, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %"'ipg321.i" = getelementptr inbounds nuw i8, ptr %24, i64 28
  %151 = getelementptr inbounds nuw i8, ptr %1, i64 28
  %152 = load float, ptr %151, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %"'ipg320.i" = getelementptr inbounds nuw i8, ptr %24, i64 24
  %153 = getelementptr inbounds nuw i8, ptr %1, i64 24
  %154 = load float, ptr %153, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %155 = fpext float %70 to double
  %"'ipg319.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 160
  %156 = getelementptr inbounds nuw i8, ptr %26, i64 160
  store double %155, ptr %156, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %157 = fpext float %154 to double
  %"'ipg316.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 168
  %158 = getelementptr inbounds nuw i8, ptr %26, i64 168
  store double %157, ptr %158, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %159 = fpext float %152 to double
  %"'ipg313.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 176
  %160 = getelementptr inbounds nuw i8, ptr %26, i64 176
  store double %159, ptr %160, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %161 = fpext float %150 to double
  %"'ipg310.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 184
  %162 = getelementptr inbounds nuw i8, ptr %26, i64 184
  store double %161, ptr %162, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %"'ipg309.i" = getelementptr inbounds nuw i8, ptr %24, i64 8
  %163 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %164 = load float, ptr %163, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %"'ipg308.i" = getelementptr inbounds nuw i8, ptr %24, i64 4
  %165 = getelementptr inbounds nuw i8, ptr %1, i64 4
  %166 = load float, ptr %165, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %167 = load float, ptr %1, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %168 = fpext float %64 to double
  %"'ipg307.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 192
  %169 = getelementptr inbounds nuw i8, ptr %26, i64 192
  store double %168, ptr %169, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %170 = fpext float %167 to double
  %"'ipg304.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 200
  %171 = getelementptr inbounds nuw i8, ptr %26, i64 200
  store double %170, ptr %171, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %172 = fpext float %166 to double
  %"'ipg301.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 208
  %173 = getelementptr inbounds nuw i8, ptr %26, i64 208
  store double %172, ptr %173, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %174 = fpext float %164 to double
  %"'ipg298.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 216
  %175 = getelementptr inbounds nuw i8, ptr %26, i64 216
  store double %174, ptr %175, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %"'ipg297.i" = getelementptr inbounds nuw i8, ptr %24, i64 20
  %176 = getelementptr inbounds nuw i8, ptr %1, i64 20
  %177 = load float, ptr %176, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %"'ipg296.i" = getelementptr inbounds nuw i8, ptr %24, i64 16
  %178 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %179 = load float, ptr %178, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %"'ipg295.i" = getelementptr inbounds nuw i8, ptr %24, i64 12
  %180 = getelementptr inbounds nuw i8, ptr %1, i64 12
  %181 = load float, ptr %180, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  tail call void @_mlir_memref_to_llvm_free(ptr %51) #8
  %182 = fpext float %67 to double
  %"'ipg294.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 224
  %183 = getelementptr inbounds nuw i8, ptr %26, i64 224
  store double %182, ptr %183, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %184 = fpext float %181 to double
  %"'ipg291.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 232
  %185 = getelementptr inbounds nuw i8, ptr %26, i64 232
  store double %184, ptr %185, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %186 = fpext float %179 to double
  %"'ipg288.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 240
  %187 = getelementptr inbounds nuw i8, ptr %26, i64 240
  store double %186, ptr %187, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %188 = fpext float %177 to double
  %"'ipg285.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 248
  %189 = getelementptr inbounds nuw i8, ptr %26, i64 248
  store double %188, ptr %189, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %190 = fpext float %44 to double
  %"'ipg282.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 256
  %191 = getelementptr inbounds nuw i8, ptr %26, i64 256
  store double %190, ptr %191, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %192 = fpext float %42 to double
  %"'ipg279.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 264
  %193 = getelementptr inbounds nuw i8, ptr %26, i64 264
  store double %192, ptr %193, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %194 = fpext float %40 to double
  %"'ipg276.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 272
  %195 = getelementptr inbounds nuw i8, ptr %26, i64 272
  store double %194, ptr %195, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %"'ipg275.i" = getelementptr inbounds nuw i8, ptr %24, i64 152
  %196 = getelementptr inbounds nuw i8, ptr %1, i64 152
  %197 = load float, ptr %196, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %"'ipg274.i" = getelementptr inbounds nuw i8, ptr %24, i64 148
  %198 = getelementptr inbounds nuw i8, ptr %1, i64 148
  %199 = load float, ptr %198, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %"'ipg273.i" = getelementptr inbounds nuw i8, ptr %24, i64 144
  %200 = getelementptr inbounds nuw i8, ptr %1, i64 144
  %201 = load float, ptr %200, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %202 = fpext float %201 to double
  %"'ipg270.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 280
  %203 = getelementptr inbounds nuw i8, ptr %26, i64 280
  store double %202, ptr %203, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %204 = fpext float %199 to double
  %"'ipg267.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 288
  %205 = getelementptr inbounds nuw i8, ptr %26, i64 288
  store double %204, ptr %205, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %206 = fpext float %197 to double
  %"'ipg264.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 296
  %207 = getelementptr inbounds nuw i8, ptr %26, i64 296
  store double %206, ptr %207, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %"'ipg263.i" = getelementptr inbounds nuw i8, ptr %24, i64 104
  %208 = getelementptr inbounds nuw i8, ptr %1, i64 104
  %209 = load float, ptr %208, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %"'ipg262.i" = getelementptr inbounds nuw i8, ptr %24, i64 100
  %210 = getelementptr inbounds nuw i8, ptr %1, i64 100
  %211 = load float, ptr %210, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %"'ipg261.i" = getelementptr inbounds nuw i8, ptr %24, i64 96
  %212 = getelementptr inbounds nuw i8, ptr %1, i64 96
  %213 = load float, ptr %212, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %214 = fpext float %213 to double
  %"'ipg258.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 304
  %215 = getelementptr inbounds nuw i8, ptr %26, i64 304
  store double %214, ptr %215, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %216 = fpext float %211 to double
  %"'ipg255.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 312
  %217 = getelementptr inbounds nuw i8, ptr %26, i64 312
  store double %216, ptr %217, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %218 = fpext float %209 to double
  %"'ipg252.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 320
  %219 = getelementptr inbounds nuw i8, ptr %26, i64 320
  store double %218, ptr %219, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %"'ipg251.i" = getelementptr inbounds nuw i8, ptr %24, i64 128
  %220 = getelementptr inbounds nuw i8, ptr %1, i64 128
  %221 = load float, ptr %220, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %"'ipg250.i" = getelementptr inbounds nuw i8, ptr %24, i64 124
  %222 = getelementptr inbounds nuw i8, ptr %1, i64 124
  %223 = load float, ptr %222, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %"'ipg249.i" = getelementptr inbounds nuw i8, ptr %24, i64 120
  %224 = getelementptr inbounds nuw i8, ptr %1, i64 120
  %225 = load float, ptr %224, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %226 = fpext float %225 to double
  %"'ipg246.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 328
  %227 = getelementptr inbounds nuw i8, ptr %26, i64 328
  store double %226, ptr %227, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %228 = fpext float %223 to double
  %"'ipg243.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 336
  %229 = getelementptr inbounds nuw i8, ptr %26, i64 336
  store double %228, ptr %229, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %230 = fpext float %221 to double
  %"'ipg240.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 344
  %231 = getelementptr inbounds nuw i8, ptr %26, i64 344
  store double %230, ptr %231, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %232 = fpext float %38 to double
  %"'ipg237.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 352
  %233 = getelementptr inbounds nuw i8, ptr %26, i64 352
  store double %232, ptr %233, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %234 = fpext float %36 to double
  %"'ipg234.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 360
  %235 = getelementptr inbounds nuw i8, ptr %26, i64 360
  store double %234, ptr %235, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %236 = fpext float %34 to double
  %"'ipg231.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 368
  %237 = getelementptr inbounds nuw i8, ptr %26, i64 368
  store double %236, ptr %237, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %"'ipg230.i" = getelementptr inbounds nuw i8, ptr %24, i64 236
  %238 = getelementptr inbounds nuw i8, ptr %1, i64 236
  %239 = load float, ptr %238, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %"'ipg229.i" = getelementptr inbounds nuw i8, ptr %24, i64 232
  %240 = getelementptr inbounds nuw i8, ptr %1, i64 232
  %241 = load float, ptr %240, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %"'ipg228.i" = getelementptr inbounds nuw i8, ptr %24, i64 228
  %242 = getelementptr inbounds nuw i8, ptr %1, i64 228
  %243 = load float, ptr %242, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %"'ipg227.i" = getelementptr inbounds nuw i8, ptr %24, i64 164
  %244 = getelementptr inbounds nuw i8, ptr %1, i64 164
  %245 = load float, ptr %244, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %"'ipg226.i" = getelementptr inbounds nuw i8, ptr %24, i64 160
  %246 = getelementptr inbounds nuw i8, ptr %1, i64 160
  %247 = load float, ptr %246, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %"'ipg225.i" = getelementptr inbounds nuw i8, ptr %24, i64 156
  %248 = getelementptr inbounds nuw i8, ptr %1, i64 156
  %249 = load float, ptr %248, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %250 = fpext float %249 to double
  %"'ipg222.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 376
  %251 = getelementptr inbounds nuw i8, ptr %26, i64 376
  store double %250, ptr %251, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %252 = fpext float %247 to double
  %"'ipg219.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 384
  %253 = getelementptr inbounds nuw i8, ptr %26, i64 384
  store double %252, ptr %253, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %254 = fpext float %245 to double
  %"'ipg216.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 392
  %255 = getelementptr inbounds nuw i8, ptr %26, i64 392
  store double %254, ptr %255, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %"'ipg215.i" = getelementptr inbounds nuw i8, ptr %24, i64 116
  %256 = getelementptr inbounds nuw i8, ptr %1, i64 116
  %257 = load float, ptr %256, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %"'ipg214.i" = getelementptr inbounds nuw i8, ptr %24, i64 112
  %258 = getelementptr inbounds nuw i8, ptr %1, i64 112
  %259 = load float, ptr %258, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %"'ipg213.i" = getelementptr inbounds nuw i8, ptr %24, i64 108
  %260 = getelementptr inbounds nuw i8, ptr %1, i64 108
  %261 = load float, ptr %260, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %262 = fpext float %261 to double
  %"'ipg210.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 400
  %263 = getelementptr inbounds nuw i8, ptr %26, i64 400
  store double %262, ptr %263, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %264 = fpext float %259 to double
  %"'ipg207.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 408
  %265 = getelementptr inbounds nuw i8, ptr %26, i64 408
  store double %264, ptr %265, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %266 = fpext float %257 to double
  %"'ipg204.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 416
  %267 = getelementptr inbounds nuw i8, ptr %26, i64 416
  store double %266, ptr %267, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %"'ipg203.i" = getelementptr inbounds nuw i8, ptr %24, i64 140
  %268 = getelementptr inbounds nuw i8, ptr %1, i64 140
  %269 = load float, ptr %268, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %"'ipg202.i" = getelementptr inbounds nuw i8, ptr %24, i64 136
  %270 = getelementptr inbounds nuw i8, ptr %1, i64 136
  %271 = load float, ptr %270, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %"'ipg201.i" = getelementptr inbounds nuw i8, ptr %24, i64 132
  %272 = getelementptr inbounds nuw i8, ptr %1, i64 132
  %273 = load float, ptr %272, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %274 = fpext float %273 to double
  %"'ipg198.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 424
  %275 = getelementptr inbounds nuw i8, ptr %26, i64 424
  store double %274, ptr %275, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %276 = fpext float %271 to double
  %"'ipg195.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 432
  %277 = getelementptr inbounds nuw i8, ptr %26, i64 432
  store double %276, ptr %277, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %278 = fpext float %269 to double
  %"'ipg192.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 440
  %279 = getelementptr inbounds nuw i8, ptr %26, i64 440
  store double %278, ptr %279, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %280 = fpext float %243 to double
  %"'ipg189.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 448
  %281 = getelementptr inbounds nuw i8, ptr %26, i64 448
  store double %280, ptr %281, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %282 = fpext float %241 to double
  %"'ipg186.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 456
  %283 = getelementptr inbounds nuw i8, ptr %26, i64 456
  store double %282, ptr %283, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %284 = fpext float %239 to double
  %"'ipg183.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 464
  %285 = getelementptr inbounds nuw i8, ptr %26, i64 464
  store double %284, ptr %285, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %"'ipg182.i" = getelementptr inbounds nuw i8, ptr %24, i64 224
  %286 = getelementptr inbounds nuw i8, ptr %1, i64 224
  %287 = load float, ptr %286, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %"'ipg181.i" = getelementptr inbounds nuw i8, ptr %24, i64 220
  %288 = getelementptr inbounds nuw i8, ptr %1, i64 220
  %289 = load float, ptr %288, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %"'ipg180.i" = getelementptr inbounds nuw i8, ptr %24, i64 216
  %290 = getelementptr inbounds nuw i8, ptr %1, i64 216
  %291 = load float, ptr %290, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %292 = fpext float %291 to double
  %"'ipg177.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 472
  %293 = getelementptr inbounds nuw i8, ptr %26, i64 472
  store double %292, ptr %293, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %294 = fpext float %289 to double
  %"'ipg174.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 480
  %295 = getelementptr inbounds nuw i8, ptr %26, i64 480
  store double %294, ptr %295, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %296 = fpext float %287 to double
  %"'ipg171.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 488
  %297 = getelementptr inbounds nuw i8, ptr %26, i64 488
  store double %296, ptr %297, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %"'ipg170.i" = getelementptr inbounds nuw i8, ptr %24, i64 260
  %298 = getelementptr inbounds nuw i8, ptr %1, i64 260
  %299 = load float, ptr %298, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %"'ipg169.i" = getelementptr inbounds nuw i8, ptr %24, i64 256
  %300 = getelementptr inbounds nuw i8, ptr %1, i64 256
  %301 = load float, ptr %300, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %"'ipg168.i" = getelementptr inbounds nuw i8, ptr %24, i64 252
  %302 = getelementptr inbounds nuw i8, ptr %1, i64 252
  %303 = load float, ptr %302, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %"'ipg167.i" = getelementptr inbounds nuw i8, ptr %24, i64 188
  %304 = getelementptr inbounds nuw i8, ptr %1, i64 188
  %305 = load float, ptr %304, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %"'ipg166.i" = getelementptr inbounds nuw i8, ptr %24, i64 184
  %306 = getelementptr inbounds nuw i8, ptr %1, i64 184
  %307 = load float, ptr %306, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %"'ipg165.i" = getelementptr inbounds nuw i8, ptr %24, i64 180
  %308 = getelementptr inbounds nuw i8, ptr %1, i64 180
  %309 = load float, ptr %308, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %310 = fpext float %309 to double
  %"'ipg162.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 496
  %311 = getelementptr inbounds nuw i8, ptr %26, i64 496
  store double %310, ptr %311, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %312 = fpext float %307 to double
  %"'ipg159.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 504
  %313 = getelementptr inbounds nuw i8, ptr %26, i64 504
  store double %312, ptr %313, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %314 = fpext float %305 to double
  %"'ipg156.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 512
  %315 = getelementptr inbounds nuw i8, ptr %26, i64 512
  store double %314, ptr %315, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %316 = fpext float %303 to double
  %"'ipg153.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 520
  %317 = getelementptr inbounds nuw i8, ptr %26, i64 520
  store double %316, ptr %317, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %318 = fpext float %301 to double
  %"'ipg150.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 528
  %319 = getelementptr inbounds nuw i8, ptr %26, i64 528
  store double %318, ptr %319, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %320 = fpext float %299 to double
  %"'ipg147.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 536
  %321 = getelementptr inbounds nuw i8, ptr %26, i64 536
  store double %320, ptr %321, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %322 = fpext float %32 to double
  %"'ipg144.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 544
  %323 = getelementptr inbounds nuw i8, ptr %26, i64 544
  store double %322, ptr %323, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %324 = fpext float %30 to double
  %"'ipg141.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 552
  %325 = getelementptr inbounds nuw i8, ptr %26, i64 552
  store double %324, ptr %325, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %326 = fpext float %28 to double
  %"'ipg138.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 560
  %327 = getelementptr inbounds nuw i8, ptr %26, i64 560
  store double %326, ptr %327, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %"'ipg137.i" = getelementptr inbounds nuw i8, ptr %24, i64 344
  %328 = getelementptr inbounds nuw i8, ptr %1, i64 344
  %329 = load float, ptr %328, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %"'ipg136.i" = getelementptr inbounds nuw i8, ptr %24, i64 340
  %330 = getelementptr inbounds nuw i8, ptr %1, i64 340
  %331 = load float, ptr %330, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %"'ipg135.i" = getelementptr inbounds nuw i8, ptr %24, i64 336
  %332 = getelementptr inbounds nuw i8, ptr %1, i64 336
  %333 = load float, ptr %332, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %"'ipg134.i" = getelementptr inbounds nuw i8, ptr %24, i64 284
  %334 = getelementptr inbounds nuw i8, ptr %1, i64 284
  %335 = load float, ptr %334, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %"'ipg133.i" = getelementptr inbounds nuw i8, ptr %24, i64 280
  %336 = getelementptr inbounds nuw i8, ptr %1, i64 280
  %337 = load float, ptr %336, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %"'ipg132.i" = getelementptr inbounds nuw i8, ptr %24, i64 276
  %338 = getelementptr inbounds nuw i8, ptr %1, i64 276
  %339 = load float, ptr %338, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %340 = fpext float %339 to double
  %"'ipg129.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 568
  %341 = getelementptr inbounds nuw i8, ptr %26, i64 568
  store double %340, ptr %341, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %342 = fpext float %337 to double
  %"'ipg126.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 576
  %343 = getelementptr inbounds nuw i8, ptr %26, i64 576
  store double %342, ptr %343, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %344 = fpext float %335 to double
  %"'ipg123.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 584
  %345 = getelementptr inbounds nuw i8, ptr %26, i64 584
  store double %344, ptr %345, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %"'ipg122.i" = getelementptr inbounds nuw i8, ptr %24, i64 212
  %346 = getelementptr inbounds nuw i8, ptr %1, i64 212
  %347 = load float, ptr %346, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %"'ipg121.i" = getelementptr inbounds nuw i8, ptr %24, i64 208
  %348 = getelementptr inbounds nuw i8, ptr %1, i64 208
  %349 = load float, ptr %348, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %"'ipg120.i" = getelementptr inbounds nuw i8, ptr %24, i64 204
  %350 = getelementptr inbounds nuw i8, ptr %1, i64 204
  %351 = load float, ptr %350, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %352 = fpext float %351 to double
  %"'ipg117.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 592
  %353 = getelementptr inbounds nuw i8, ptr %26, i64 592
  store double %352, ptr %353, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %354 = fpext float %349 to double
  %"'ipg114.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 600
  %355 = getelementptr inbounds nuw i8, ptr %26, i64 600
  store double %354, ptr %355, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %356 = fpext float %347 to double
  %"'ipg111.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 608
  %357 = getelementptr inbounds nuw i8, ptr %26, i64 608
  store double %356, ptr %357, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %"'ipg110.i" = getelementptr inbounds nuw i8, ptr %24, i64 248
  %358 = getelementptr inbounds nuw i8, ptr %1, i64 248
  %359 = load float, ptr %358, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %"'ipg109.i" = getelementptr inbounds nuw i8, ptr %24, i64 244
  %360 = getelementptr inbounds nuw i8, ptr %1, i64 244
  %361 = load float, ptr %360, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %"'ipg108.i" = getelementptr inbounds nuw i8, ptr %24, i64 240
  %362 = getelementptr inbounds nuw i8, ptr %1, i64 240
  %363 = load float, ptr %362, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %364 = fpext float %363 to double
  %"'ipg105.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 616
  %365 = getelementptr inbounds nuw i8, ptr %26, i64 616
  store double %364, ptr %365, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %366 = fpext float %361 to double
  %"'ipg102.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 624
  %367 = getelementptr inbounds nuw i8, ptr %26, i64 624
  store double %366, ptr %367, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %368 = fpext float %359 to double
  %"'ipg99.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 632
  %369 = getelementptr inbounds nuw i8, ptr %26, i64 632
  store double %368, ptr %369, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %370 = fpext float %333 to double
  %"'ipg96.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 640
  %371 = getelementptr inbounds nuw i8, ptr %26, i64 640
  store double %370, ptr %371, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %372 = fpext float %331 to double
  %"'ipg93.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 648
  %373 = getelementptr inbounds nuw i8, ptr %26, i64 648
  store double %372, ptr %373, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %374 = fpext float %329 to double
  %"'ipg90.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 656
  %375 = getelementptr inbounds nuw i8, ptr %26, i64 656
  store double %374, ptr %375, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %"'ipg89.i" = getelementptr inbounds nuw i8, ptr %24, i64 308
  %376 = getelementptr inbounds nuw i8, ptr %1, i64 308
  %377 = load float, ptr %376, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %"'ipg88.i" = getelementptr inbounds nuw i8, ptr %24, i64 304
  %378 = getelementptr inbounds nuw i8, ptr %1, i64 304
  %379 = load float, ptr %378, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %"'ipg87.i" = getelementptr inbounds nuw i8, ptr %24, i64 300
  %380 = getelementptr inbounds nuw i8, ptr %1, i64 300
  %381 = load float, ptr %380, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %"'ipg86.i" = getelementptr inbounds nuw i8, ptr %24, i64 272
  %382 = getelementptr inbounds nuw i8, ptr %1, i64 272
  %383 = load float, ptr %382, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %"'ipg85.i" = getelementptr inbounds nuw i8, ptr %24, i64 268
  %384 = getelementptr inbounds nuw i8, ptr %1, i64 268
  %385 = load float, ptr %384, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %"'ipg84.i" = getelementptr inbounds nuw i8, ptr %24, i64 264
  %386 = getelementptr inbounds nuw i8, ptr %1, i64 264
  %387 = load float, ptr %386, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %388 = fpext float %387 to double
  %"'ipg81.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 664
  %389 = getelementptr inbounds nuw i8, ptr %26, i64 664
  store double %388, ptr %389, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %390 = fpext float %385 to double
  %"'ipg78.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 672
  %391 = getelementptr inbounds nuw i8, ptr %26, i64 672
  store double %390, ptr %391, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %392 = fpext float %383 to double
  %"'ipg75.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 680
  %393 = getelementptr inbounds nuw i8, ptr %26, i64 680
  store double %392, ptr %393, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %394 = fpext float %381 to double
  %"'ipg72.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 688
  %395 = getelementptr inbounds nuw i8, ptr %26, i64 688
  store double %394, ptr %395, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %396 = fpext float %379 to double
  %"'ipg69.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 696
  %397 = getelementptr inbounds nuw i8, ptr %26, i64 696
  store double %396, ptr %397, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %398 = fpext float %377 to double
  %"'ipg66.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 704
  %399 = getelementptr inbounds nuw i8, ptr %26, i64 704
  store double %398, ptr %399, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %"'ipg65.i" = getelementptr inbounds nuw i8, ptr %24, i64 356
  %400 = getelementptr inbounds nuw i8, ptr %1, i64 356
  %401 = load float, ptr %400, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %"'ipg64.i" = getelementptr inbounds nuw i8, ptr %24, i64 352
  %402 = getelementptr inbounds nuw i8, ptr %1, i64 352
  %403 = load float, ptr %402, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %"'ipg63.i" = getelementptr inbounds nuw i8, ptr %24, i64 348
  %404 = getelementptr inbounds nuw i8, ptr %1, i64 348
  %405 = load float, ptr %404, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %406 = fpext float %405 to double
  %"'ipg60.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 712
  %407 = getelementptr inbounds nuw i8, ptr %26, i64 712
  store double %406, ptr %407, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %408 = fpext float %403 to double
  %"'ipg57.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 720
  %409 = getelementptr inbounds nuw i8, ptr %26, i64 720
  store double %408, ptr %409, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %410 = fpext float %401 to double
  %"'ipg54.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 728
  %411 = getelementptr inbounds nuw i8, ptr %26, i64 728
  store double %410, ptr %411, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %"'ipg53.i" = getelementptr inbounds nuw i8, ptr %24, i64 320
  %412 = getelementptr inbounds nuw i8, ptr %1, i64 320
  %413 = load float, ptr %412, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %"'ipg52.i" = getelementptr inbounds nuw i8, ptr %24, i64 316
  %414 = getelementptr inbounds nuw i8, ptr %1, i64 316
  %415 = load float, ptr %414, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %"'ipg51.i" = getelementptr inbounds nuw i8, ptr %24, i64 312
  %416 = getelementptr inbounds nuw i8, ptr %1, i64 312
  %417 = load float, ptr %416, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %418 = fpext float %417 to double
  %"'ipg48.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 736
  %419 = getelementptr inbounds nuw i8, ptr %26, i64 736
  store double %418, ptr %419, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %420 = fpext float %415 to double
  %"'ipg45.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 744
  %421 = getelementptr inbounds nuw i8, ptr %26, i64 744
  store double %420, ptr %421, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %422 = fpext float %413 to double
  %"'ipg42.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 752
  %423 = getelementptr inbounds nuw i8, ptr %26, i64 752
  store double %422, ptr %423, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %"'ipg41.i" = getelementptr inbounds nuw i8, ptr %24, i64 368
  %424 = getelementptr inbounds nuw i8, ptr %1, i64 368
  %425 = load float, ptr %424, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %"'ipg40.i" = getelementptr inbounds nuw i8, ptr %24, i64 364
  %426 = getelementptr inbounds nuw i8, ptr %1, i64 364
  %427 = load float, ptr %426, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %"'ipg39.i" = getelementptr inbounds nuw i8, ptr %24, i64 360
  %428 = getelementptr inbounds nuw i8, ptr %1, i64 360
  %429 = load float, ptr %428, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %430 = fpext float %429 to double
  %"'ipg36.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 760
  %431 = getelementptr inbounds nuw i8, ptr %26, i64 760
  store double %430, ptr %431, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %432 = fpext float %427 to double
  %"'ipg33.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 768
  %433 = getelementptr inbounds nuw i8, ptr %26, i64 768
  store double %432, ptr %433, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %434 = fpext float %425 to double
  %"'ipg30.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 776
  %435 = getelementptr inbounds nuw i8, ptr %26, i64 776
  store double %434, ptr %435, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %"'ipg29.i" = getelementptr inbounds nuw i8, ptr %24, i64 332
  %436 = getelementptr inbounds nuw i8, ptr %1, i64 332
  %437 = load float, ptr %436, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %"'ipg28.i" = getelementptr inbounds nuw i8, ptr %24, i64 328
  %438 = getelementptr inbounds nuw i8, ptr %1, i64 328
  %439 = load float, ptr %438, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %"'ipg27.i" = getelementptr inbounds nuw i8, ptr %24, i64 324
  %440 = getelementptr inbounds nuw i8, ptr %1, i64 324
  %441 = load float, ptr %440, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %442 = fpext float %441 to double
  %"'ipg24.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 784
  %443 = getelementptr inbounds nuw i8, ptr %26, i64 784
  store double %442, ptr %443, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %444 = fpext float %439 to double
  %"'ipg21.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 792
  %445 = getelementptr inbounds nuw i8, ptr %26, i64 792
  store double %444, ptr %445, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %446 = fpext float %437 to double
  %"'ipg18.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 800
  %447 = getelementptr inbounds nuw i8, ptr %26, i64 800
  store double %446, ptr %447, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %"'ipg17.i" = getelementptr inbounds nuw i8, ptr %24, i64 380
  %448 = getelementptr inbounds nuw i8, ptr %1, i64 380
  %449 = load float, ptr %448, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %"'ipg16.i" = getelementptr inbounds nuw i8, ptr %24, i64 376
  %450 = getelementptr inbounds nuw i8, ptr %1, i64 376
  %451 = load float, ptr %450, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %"'ipg15.i" = getelementptr inbounds nuw i8, ptr %24, i64 372
  %452 = getelementptr inbounds nuw i8, ptr %1, i64 372
  %453 = load float, ptr %452, align 4, !tbaa !5, !alias.scope !7, !noalias !10
  %454 = fpext float %453 to double
  %"'ipg12.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 808
  %455 = getelementptr inbounds nuw i8, ptr %26, i64 808
  store double %454, ptr %455, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %456 = fpext float %451 to double
  %"'ipg9.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 816
  %457 = getelementptr inbounds nuw i8, ptr %26, i64 816
  store double %456, ptr %457, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  %458 = fpext float %449 to double
  %"'ipg.i" = getelementptr inbounds nuw i8, ptr %"'mi.i", i64 824
  %459 = getelementptr inbounds nuw i8, ptr %26, i64 824
  store double %458, ptr %459, align 8, !tbaa !2, !alias.scope !17, !noalias !20
  store ptr %0, ptr %18, align 8, !alias.scope !22, !noalias !25
  %.fca.1.gep.i = getelementptr inbounds nuw i8, ptr %18, i64 8
  store ptr %1, ptr %.fca.1.gep.i, align 8, !alias.scope !22, !noalias !25
  %.fca.2.gep.i = getelementptr inbounds nuw i8, ptr %18, i64 16
  store i64 %2, ptr %.fca.2.gep.i, align 8, !alias.scope !22, !noalias !25
  %.fca.3.0.gep.i = getelementptr inbounds nuw i8, ptr %18, i64 24
  store i64 %3, ptr %.fca.3.0.gep.i, align 8, !alias.scope !22, !noalias !25
  %.fca.3.1.gep.i = getelementptr inbounds nuw i8, ptr %18, i64 32
  store i64 %4, ptr %.fca.3.1.gep.i, align 8, !alias.scope !22, !noalias !25
  %.fca.3.2.gep.i = getelementptr inbounds nuw i8, ptr %18, i64 40
  store i64 %5, ptr %.fca.3.2.gep.i, align 8, !alias.scope !22, !noalias !25
  %.fca.4.0.gep.i = getelementptr inbounds nuw i8, ptr %18, i64 48
  store i64 %6, ptr %.fca.4.0.gep.i, align 8, !alias.scope !22, !noalias !25
  %.fca.4.1.gep.i = getelementptr inbounds nuw i8, ptr %18, i64 56
  store i64 %7, ptr %.fca.4.1.gep.i, align 8, !alias.scope !22, !noalias !25
  %.fca.4.2.gep.i = getelementptr inbounds nuw i8, ptr %18, i64 64
  store i64 %8, ptr %.fca.4.2.gep.i, align 8, !alias.scope !22, !noalias !25
  store ptr %9, ptr %17, align 8, !alias.scope !27, !noalias !30
  %.fca.1.gep107.i = getelementptr inbounds nuw i8, ptr %17, i64 8
  store ptr %10, ptr %.fca.1.gep107.i, align 8, !alias.scope !27, !noalias !30
  %.fca.2.gep109.i = getelementptr inbounds nuw i8, ptr %17, i64 16
  store i64 %11, ptr %.fca.2.gep109.i, align 8, !alias.scope !27, !noalias !30
  %.fca.3.0.gep111.i = getelementptr inbounds nuw i8, ptr %17, i64 24
  store i64 %12, ptr %.fca.3.0.gep111.i, align 8, !alias.scope !27, !noalias !30
  %.fca.4.0.gep113.i = getelementptr inbounds nuw i8, ptr %17, i64 32
  store i64 %13, ptr %.fca.4.0.gep113.i, align 8, !alias.scope !27, !noalias !30
  store ptr %"'mi.i", ptr %"'ipa4.i", align 8, !alias.scope !32, !noalias !35
  store ptr %26, ptr %16, align 8, !alias.scope !35, !noalias !32
  %.fca.1.gep117.i = getelementptr inbounds nuw i8, ptr %16, i64 8
  store ptr %"'mi.i", ptr %.fca.1.gep410.i, align 8, !alias.scope !32, !noalias !35
  store ptr %26, ptr %.fca.1.gep117.i, align 8, !alias.scope !35, !noalias !32
  %.fca.2.gep119.i = getelementptr inbounds nuw i8, ptr %16, i64 16
  store i64 0, ptr %.fca.2.gep411.i, align 8, !alias.scope !32, !noalias !35
  store i64 0, ptr %.fca.2.gep119.i, align 8, !alias.scope !35, !noalias !32
  %.fca.3.0.gep121.i = getelementptr inbounds nuw i8, ptr %16, i64 24
  store i64 104, ptr %.fca.3.0.gep412.i, align 8, !alias.scope !32, !noalias !35
  store i64 104, ptr %.fca.3.0.gep121.i, align 8, !alias.scope !35, !noalias !32
  %.fca.4.0.gep123.i = getelementptr inbounds nuw i8, ptr %16, i64 32
  store i64 1, ptr %.fca.4.0.gep413.i, align 8, !alias.scope !32, !noalias !35
  store i64 1, ptr %.fca.4.0.gep123.i, align 8, !alias.scope !35, !noalias !32
  %"'mi6.i" = tail call noalias nonnull ptr @_mlir_memref_to_llvm_alloc(i64 8) #8
  store i64 0, ptr %"'mi6.i", align 1
  %460 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 8) #8
  store ptr %"'mi6.i", ptr %"'ipa5.i", align 8, !alias.scope !37, !noalias !40
  store ptr %460, ptr %15, align 8, !alias.scope !40, !noalias !37
  %.fca.1.gep127.i = getelementptr inbounds nuw i8, ptr %15, i64 8
  store ptr %"'mi6.i", ptr %.fca.1.gep415.i, align 8, !alias.scope !37, !noalias !40
  store ptr %460, ptr %.fca.1.gep127.i, align 8, !alias.scope !40, !noalias !37
  %.fca.2.gep129.i = getelementptr inbounds nuw i8, ptr %15, i64 16
  store i64 0, ptr %.fca.2.gep416.i, align 8, !alias.scope !37, !noalias !40
  store i64 0, ptr %.fca.2.gep129.i, align 8, !alias.scope !40, !noalias !37
  %461 = load double, ptr %23, align 64, !tbaa !2, !alias.scope !42, !noalias !45
  store double 0.000000e+00, ptr %23, align 64, !tbaa !2, !alias.scope !42, !noalias !45
  store double %461, ptr %"'mi6.i", align 8, !tbaa !2, !alias.scope !47, !noalias !50
  call void @qnode_forward_0.quantum(ptr nonnull %18, ptr nonnull %17, ptr nonnull %16, ptr nonnull %15)
  call void @qnode_forward_0.quantum.customqgrad(ptr nonnull %18, ptr nonnull readnone poison, ptr nonnull %17, ptr nonnull readnone poison, ptr nonnull poison, ptr nonnull readonly %"'ipa4.i", ptr nonnull poison, ptr nonnull readonly %"'ipa5.i", ptr poison)
  call void @_mlir_memref_to_llvm_free(ptr nonnull %"'mi6.i")
  call void @_mlir_memref_to_llvm_free(ptr %460)
  %462 = load double, ptr %"'ipg.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %463 = fptrunc fast double %462 to float
  %464 = load double, ptr %"'ipg9.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg9.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %465 = fptrunc fast double %464 to float
  %466 = load double, ptr %"'ipg12.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg12.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %467 = fptrunc fast double %466 to float
  %468 = load float, ptr %"'ipg15.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %469 = fadd fast float %468, %467
  store float %469, ptr %"'ipg15.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %470 = load float, ptr %"'ipg16.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %471 = fadd fast float %470, %465
  store float %471, ptr %"'ipg16.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %472 = load float, ptr %"'ipg17.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %473 = fadd fast float %472, %463
  store float %473, ptr %"'ipg17.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %474 = load double, ptr %"'ipg18.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg18.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %475 = fptrunc fast double %474 to float
  %476 = load double, ptr %"'ipg21.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg21.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %477 = fptrunc fast double %476 to float
  %478 = load double, ptr %"'ipg24.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg24.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %479 = fptrunc fast double %478 to float
  %480 = load float, ptr %"'ipg27.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %481 = fadd fast float %480, %479
  store float %481, ptr %"'ipg27.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %482 = load float, ptr %"'ipg28.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %483 = fadd fast float %482, %477
  store float %483, ptr %"'ipg28.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %484 = load float, ptr %"'ipg29.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %485 = fadd fast float %484, %475
  store float %485, ptr %"'ipg29.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %486 = load double, ptr %"'ipg30.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg30.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %487 = fptrunc fast double %486 to float
  %488 = load double, ptr %"'ipg33.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg33.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %489 = fptrunc fast double %488 to float
  %490 = load double, ptr %"'ipg36.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg36.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %491 = fptrunc fast double %490 to float
  %492 = load float, ptr %"'ipg39.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %493 = fadd fast float %492, %491
  store float %493, ptr %"'ipg39.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %494 = load float, ptr %"'ipg40.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %495 = fadd fast float %494, %489
  store float %495, ptr %"'ipg40.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %496 = load float, ptr %"'ipg41.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %497 = fadd fast float %496, %487
  store float %497, ptr %"'ipg41.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %498 = load double, ptr %"'ipg42.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg42.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %499 = fptrunc fast double %498 to float
  %500 = load double, ptr %"'ipg45.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg45.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %501 = fptrunc fast double %500 to float
  %502 = load double, ptr %"'ipg48.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg48.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %503 = fptrunc fast double %502 to float
  %504 = load float, ptr %"'ipg51.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %505 = fadd fast float %504, %503
  store float %505, ptr %"'ipg51.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %506 = load float, ptr %"'ipg52.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %507 = fadd fast float %506, %501
  store float %507, ptr %"'ipg52.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %508 = load float, ptr %"'ipg53.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %509 = fadd fast float %508, %499
  store float %509, ptr %"'ipg53.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %510 = load double, ptr %"'ipg54.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg54.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %511 = fptrunc fast double %510 to float
  %512 = load double, ptr %"'ipg57.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg57.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %513 = fptrunc fast double %512 to float
  %514 = load double, ptr %"'ipg60.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg60.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %515 = fptrunc fast double %514 to float
  %516 = load float, ptr %"'ipg63.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %517 = fadd fast float %516, %515
  store float %517, ptr %"'ipg63.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %518 = load float, ptr %"'ipg64.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %519 = fadd fast float %518, %513
  store float %519, ptr %"'ipg64.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %520 = load float, ptr %"'ipg65.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %521 = fadd fast float %520, %511
  store float %521, ptr %"'ipg65.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %522 = load double, ptr %"'ipg66.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg66.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %523 = fptrunc fast double %522 to float
  %524 = load double, ptr %"'ipg69.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg69.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %525 = fptrunc fast double %524 to float
  %526 = load double, ptr %"'ipg72.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg72.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %527 = fptrunc fast double %526 to float
  %528 = load double, ptr %"'ipg75.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg75.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %529 = fptrunc fast double %528 to float
  %530 = load double, ptr %"'ipg78.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg78.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %531 = fptrunc fast double %530 to float
  %532 = load double, ptr %"'ipg81.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg81.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %533 = fptrunc fast double %532 to float
  %534 = load float, ptr %"'ipg84.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %535 = fadd fast float %534, %533
  store float %535, ptr %"'ipg84.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %536 = load float, ptr %"'ipg85.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %537 = fadd fast float %536, %531
  store float %537, ptr %"'ipg85.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %538 = load float, ptr %"'ipg86.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %539 = fadd fast float %538, %529
  store float %539, ptr %"'ipg86.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %540 = load float, ptr %"'ipg87.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %541 = fadd fast float %540, %527
  store float %541, ptr %"'ipg87.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %542 = load float, ptr %"'ipg88.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %543 = fadd fast float %542, %525
  store float %543, ptr %"'ipg88.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %544 = load float, ptr %"'ipg89.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %545 = fadd fast float %544, %523
  store float %545, ptr %"'ipg89.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %546 = load double, ptr %"'ipg90.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg90.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %547 = fptrunc fast double %546 to float
  %548 = load double, ptr %"'ipg93.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg93.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %549 = fptrunc fast double %548 to float
  %550 = load double, ptr %"'ipg96.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg96.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %551 = fptrunc fast double %550 to float
  %552 = load double, ptr %"'ipg99.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg99.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %553 = fptrunc fast double %552 to float
  %554 = load double, ptr %"'ipg102.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg102.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %555 = fptrunc fast double %554 to float
  %556 = load double, ptr %"'ipg105.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg105.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %557 = fptrunc fast double %556 to float
  %558 = load float, ptr %"'ipg108.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %559 = fadd fast float %558, %557
  store float %559, ptr %"'ipg108.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %560 = load float, ptr %"'ipg109.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %561 = fadd fast float %560, %555
  store float %561, ptr %"'ipg109.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %562 = load float, ptr %"'ipg110.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %563 = fadd fast float %562, %553
  store float %563, ptr %"'ipg110.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %564 = load double, ptr %"'ipg111.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg111.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %565 = fptrunc fast double %564 to float
  %566 = load double, ptr %"'ipg114.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg114.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %567 = fptrunc fast double %566 to float
  %568 = load double, ptr %"'ipg117.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg117.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %569 = fptrunc fast double %568 to float
  %570 = load float, ptr %"'ipg120.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %571 = fadd fast float %570, %569
  store float %571, ptr %"'ipg120.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %572 = load float, ptr %"'ipg121.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %573 = fadd fast float %572, %567
  store float %573, ptr %"'ipg121.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %574 = load float, ptr %"'ipg122.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %575 = fadd fast float %574, %565
  store float %575, ptr %"'ipg122.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %576 = load double, ptr %"'ipg123.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg123.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %577 = fptrunc fast double %576 to float
  %578 = load double, ptr %"'ipg126.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg126.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %579 = fptrunc fast double %578 to float
  %580 = load double, ptr %"'ipg129.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg129.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %581 = fptrunc fast double %580 to float
  %582 = load float, ptr %"'ipg132.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %583 = fadd fast float %582, %581
  store float %583, ptr %"'ipg132.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %584 = load float, ptr %"'ipg133.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %585 = fadd fast float %584, %579
  store float %585, ptr %"'ipg133.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %586 = load float, ptr %"'ipg134.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %587 = fadd fast float %586, %577
  store float %587, ptr %"'ipg134.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %588 = load float, ptr %"'ipg135.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %589 = fadd fast float %588, %551
  store float %589, ptr %"'ipg135.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %590 = load float, ptr %"'ipg136.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %591 = fadd fast float %590, %549
  store float %591, ptr %"'ipg136.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %592 = load float, ptr %"'ipg137.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %593 = fadd fast float %592, %547
  store float %593, ptr %"'ipg137.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %594 = load double, ptr %"'ipg138.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg138.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %595 = fptrunc fast double %594 to float
  %596 = load double, ptr %"'ipg141.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg141.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %597 = fptrunc fast double %596 to float
  %598 = load double, ptr %"'ipg144.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg144.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %599 = fptrunc fast double %598 to float
  %600 = load double, ptr %"'ipg147.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg147.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %601 = fptrunc fast double %600 to float
  %602 = load double, ptr %"'ipg150.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg150.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %603 = fptrunc fast double %602 to float
  %604 = load double, ptr %"'ipg153.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg153.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %605 = fptrunc fast double %604 to float
  %606 = load double, ptr %"'ipg156.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg156.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %607 = fptrunc fast double %606 to float
  %608 = load double, ptr %"'ipg159.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg159.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %609 = fptrunc fast double %608 to float
  %610 = load double, ptr %"'ipg162.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg162.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %611 = fptrunc fast double %610 to float
  %612 = load float, ptr %"'ipg165.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %613 = fadd fast float %612, %611
  store float %613, ptr %"'ipg165.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %614 = load float, ptr %"'ipg166.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %615 = fadd fast float %614, %609
  store float %615, ptr %"'ipg166.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %616 = load float, ptr %"'ipg167.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %617 = fadd fast float %616, %607
  store float %617, ptr %"'ipg167.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %618 = load float, ptr %"'ipg168.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %619 = fadd fast float %618, %605
  store float %619, ptr %"'ipg168.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %620 = load float, ptr %"'ipg169.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %621 = fadd fast float %620, %603
  store float %621, ptr %"'ipg169.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %622 = load float, ptr %"'ipg170.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %623 = fadd fast float %622, %601
  store float %623, ptr %"'ipg170.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %624 = load double, ptr %"'ipg171.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg171.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %625 = fptrunc fast double %624 to float
  %626 = load double, ptr %"'ipg174.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg174.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %627 = fptrunc fast double %626 to float
  %628 = load double, ptr %"'ipg177.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg177.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %629 = fptrunc fast double %628 to float
  %630 = load float, ptr %"'ipg180.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %631 = fadd fast float %630, %629
  store float %631, ptr %"'ipg180.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %632 = load float, ptr %"'ipg181.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %633 = fadd fast float %632, %627
  store float %633, ptr %"'ipg181.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %634 = load float, ptr %"'ipg182.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %635 = fadd fast float %634, %625
  store float %635, ptr %"'ipg182.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %636 = load double, ptr %"'ipg183.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg183.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %637 = fptrunc fast double %636 to float
  %638 = load double, ptr %"'ipg186.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg186.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %639 = fptrunc fast double %638 to float
  %640 = load double, ptr %"'ipg189.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg189.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %641 = fptrunc fast double %640 to float
  %642 = load double, ptr %"'ipg192.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg192.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %643 = fptrunc fast double %642 to float
  %644 = load double, ptr %"'ipg195.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg195.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %645 = fptrunc fast double %644 to float
  %646 = load double, ptr %"'ipg198.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg198.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %647 = fptrunc fast double %646 to float
  %648 = load float, ptr %"'ipg201.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %649 = fadd fast float %648, %647
  store float %649, ptr %"'ipg201.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %650 = load float, ptr %"'ipg202.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %651 = fadd fast float %650, %645
  store float %651, ptr %"'ipg202.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %652 = load float, ptr %"'ipg203.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %653 = fadd fast float %652, %643
  store float %653, ptr %"'ipg203.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %654 = load double, ptr %"'ipg204.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg204.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %655 = fptrunc fast double %654 to float
  %656 = load double, ptr %"'ipg207.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg207.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %657 = fptrunc fast double %656 to float
  %658 = load double, ptr %"'ipg210.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg210.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %659 = fptrunc fast double %658 to float
  %660 = load float, ptr %"'ipg213.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %661 = fadd fast float %660, %659
  store float %661, ptr %"'ipg213.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %662 = load float, ptr %"'ipg214.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %663 = fadd fast float %662, %657
  store float %663, ptr %"'ipg214.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %664 = load float, ptr %"'ipg215.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %665 = fadd fast float %664, %655
  store float %665, ptr %"'ipg215.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %666 = load double, ptr %"'ipg216.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg216.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %667 = fptrunc fast double %666 to float
  %668 = load double, ptr %"'ipg219.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg219.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %669 = fptrunc fast double %668 to float
  %670 = load double, ptr %"'ipg222.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg222.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %671 = fptrunc fast double %670 to float
  %672 = load float, ptr %"'ipg225.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %673 = fadd fast float %672, %671
  store float %673, ptr %"'ipg225.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %674 = load float, ptr %"'ipg226.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %675 = fadd fast float %674, %669
  store float %675, ptr %"'ipg226.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %676 = load float, ptr %"'ipg227.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %677 = fadd fast float %676, %667
  store float %677, ptr %"'ipg227.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %678 = load float, ptr %"'ipg228.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %679 = fadd fast float %678, %641
  store float %679, ptr %"'ipg228.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %680 = load float, ptr %"'ipg229.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %681 = fadd fast float %680, %639
  store float %681, ptr %"'ipg229.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %682 = load float, ptr %"'ipg230.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %683 = fadd fast float %682, %637
  store float %683, ptr %"'ipg230.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %684 = load double, ptr %"'ipg231.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg231.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %685 = fptrunc fast double %684 to float
  %686 = load double, ptr %"'ipg234.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg234.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %687 = fptrunc fast double %686 to float
  %688 = load double, ptr %"'ipg237.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg237.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %689 = fptrunc fast double %688 to float
  %690 = load double, ptr %"'ipg240.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg240.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %691 = fptrunc fast double %690 to float
  %692 = load double, ptr %"'ipg243.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg243.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %693 = fptrunc fast double %692 to float
  %694 = load double, ptr %"'ipg246.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg246.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %695 = fptrunc fast double %694 to float
  %696 = load float, ptr %"'ipg249.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %697 = fadd fast float %696, %695
  store float %697, ptr %"'ipg249.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %698 = load float, ptr %"'ipg250.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %699 = fadd fast float %698, %693
  store float %699, ptr %"'ipg250.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %700 = load float, ptr %"'ipg251.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %701 = fadd fast float %700, %691
  store float %701, ptr %"'ipg251.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %702 = load double, ptr %"'ipg252.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg252.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %703 = fptrunc fast double %702 to float
  %704 = load double, ptr %"'ipg255.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg255.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %705 = fptrunc fast double %704 to float
  %706 = load double, ptr %"'ipg258.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg258.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %707 = fptrunc fast double %706 to float
  %708 = load float, ptr %"'ipg261.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %709 = fadd fast float %708, %707
  store float %709, ptr %"'ipg261.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %710 = load float, ptr %"'ipg262.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %711 = fadd fast float %710, %705
  store float %711, ptr %"'ipg262.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %712 = load float, ptr %"'ipg263.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %713 = fadd fast float %712, %703
  store float %713, ptr %"'ipg263.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %714 = load double, ptr %"'ipg264.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg264.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %715 = fptrunc fast double %714 to float
  %716 = load double, ptr %"'ipg267.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg267.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %717 = fptrunc fast double %716 to float
  %718 = load double, ptr %"'ipg270.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg270.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %719 = fptrunc fast double %718 to float
  %720 = load float, ptr %"'ipg273.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %721 = fadd fast float %720, %719
  store float %721, ptr %"'ipg273.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %722 = load float, ptr %"'ipg274.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %723 = fadd fast float %722, %717
  store float %723, ptr %"'ipg274.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %724 = load float, ptr %"'ipg275.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %725 = fadd fast float %724, %715
  store float %725, ptr %"'ipg275.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %726 = load double, ptr %"'ipg276.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg276.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %727 = fptrunc fast double %726 to float
  %728 = load double, ptr %"'ipg279.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg279.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %729 = fptrunc fast double %728 to float
  %730 = load double, ptr %"'ipg282.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg282.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %731 = fptrunc fast double %730 to float
  %732 = load double, ptr %"'ipg285.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg285.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %733 = fptrunc fast double %732 to float
  %734 = load double, ptr %"'ipg288.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg288.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %735 = fptrunc fast double %734 to float
  %736 = load double, ptr %"'ipg291.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %737 = fptrunc fast double %736 to float
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %"'ipg294.i", i8 0, i64 16, i1 false)
  %738 = load float, ptr %"'ipg295.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %739 = fadd fast float %738, %737
  store float %739, ptr %"'ipg295.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %740 = load float, ptr %"'ipg296.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %741 = fadd fast float %740, %735
  store float %741, ptr %"'ipg296.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %742 = load float, ptr %"'ipg297.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %743 = fadd fast float %742, %733
  store float %743, ptr %"'ipg297.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %744 = load double, ptr %"'ipg298.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg298.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %745 = fptrunc fast double %744 to float
  %746 = load double, ptr %"'ipg301.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg301.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %747 = fptrunc fast double %746 to float
  %748 = load double, ptr %"'ipg304.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %749 = fptrunc fast double %748 to float
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %"'ipg307.i", i8 0, i64 16, i1 false)
  %750 = load float, ptr %24, align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %751 = fadd fast float %750, %749
  store float %751, ptr %24, align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %752 = load float, ptr %"'ipg308.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %753 = fadd fast float %752, %747
  store float %753, ptr %"'ipg308.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %754 = load float, ptr %"'ipg309.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %755 = fadd fast float %754, %745
  store float %755, ptr %"'ipg309.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %756 = load double, ptr %"'ipg310.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg310.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %757 = fptrunc fast double %756 to float
  %758 = load double, ptr %"'ipg313.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg313.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %759 = fptrunc fast double %758 to float
  %760 = load double, ptr %"'ipg316.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %761 = fptrunc fast double %760 to float
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %"'ipg319.i", i8 0, i64 16, i1 false)
  %762 = load float, ptr %"'ipg320.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %763 = fadd fast float %762, %761
  store float %763, ptr %"'ipg320.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %764 = load float, ptr %"'ipg321.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %765 = fadd fast float %764, %759
  store float %765, ptr %"'ipg321.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %766 = load float, ptr %"'ipg322.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %767 = fadd fast float %766, %757
  store float %767, ptr %"'ipg322.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %768 = load double, ptr %"'ipg323.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg323.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %769 = fptrunc fast double %768 to float
  %770 = load double, ptr %"'ipg326.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg326.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %771 = fptrunc fast double %770 to float
  %772 = load double, ptr %"'ipg329.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %773 = fptrunc fast double %772 to float
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %"'ipg332.i", i8 0, i64 16, i1 false)
  %774 = load float, ptr %"'ipg333.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %775 = fadd fast float %774, %773
  store float %775, ptr %"'ipg333.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %776 = load float, ptr %"'ipg334.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %777 = fadd fast float %776, %771
  store float %777, ptr %"'ipg334.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %778 = load float, ptr %"'ipg335.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %779 = fadd fast float %778, %769
  store float %779, ptr %"'ipg335.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %780 = load double, ptr %"'ipg336.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg336.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %781 = fptrunc fast double %780 to float
  %782 = load double, ptr %"'ipg339.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg339.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %783 = fptrunc fast double %782 to float
  %784 = load double, ptr %"'ipg342.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %785 = fptrunc fast double %784 to float
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %"'ipg345.i", i8 0, i64 16, i1 false)
  %786 = load float, ptr %"'ipg346.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %787 = fadd fast float %786, %785
  store float %787, ptr %"'ipg346.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %788 = load float, ptr %"'ipg347.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %789 = fadd fast float %788, %783
  store float %789, ptr %"'ipg347.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %790 = load float, ptr %"'ipg348.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %791 = fadd fast float %790, %781
  store float %791, ptr %"'ipg348.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %792 = load double, ptr %"'ipg349.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg349.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %793 = fptrunc fast double %792 to float
  %794 = load double, ptr %"'ipg352.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg352.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %795 = fptrunc fast double %794 to float
  %796 = load double, ptr %"'ipg355.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %797 = fptrunc fast double %796 to float
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %"'ipg358.i", i8 0, i64 16, i1 false)
  %798 = load float, ptr %"'ipg359.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %799 = fadd fast float %798, %797
  store float %799, ptr %"'ipg359.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %800 = load float, ptr %"'ipg360.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %801 = fadd fast float %800, %795
  store float %801, ptr %"'ipg360.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %802 = load float, ptr %"'ipg361.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %803 = fadd fast float %802, %793
  store float %803, ptr %"'ipg361.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %804 = load double, ptr %"'ipg362.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg362.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %805 = fptrunc fast double %804 to float
  %806 = load double, ptr %"'ipg365.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg365.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %807 = fptrunc fast double %806 to float
  %808 = load double, ptr %"'ipg368.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %809 = fptrunc fast double %808 to float
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %"'ipg371.i", i8 0, i64 16, i1 false)
  %810 = load float, ptr %"'ipg372.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %811 = fadd fast float %810, %809
  store float %811, ptr %"'ipg372.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %812 = load float, ptr %"'ipg373.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %813 = fadd fast float %812, %807
  store float %813, ptr %"'ipg373.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %814 = load float, ptr %"'ipg374.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %815 = fadd fast float %814, %805
  store float %815, ptr %"'ipg374.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %816 = load double, ptr %"'ipg375.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg375.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %817 = fptrunc fast double %816 to float
  %818 = load double, ptr %"'ipg378.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  store double 0.000000e+00, ptr %"'ipg378.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %819 = fptrunc fast double %818 to float
  %820 = load double, ptr %"'ipg381.i", align 8, !tbaa !2, !alias.scope !20, !noalias !17
  %821 = fptrunc fast double %820 to float
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %"'mi.i", i8 0, i64 16, i1 false)
  %822 = load float, ptr %"'ipg384.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %823 = fadd fast float %822, %821
  store float %823, ptr %"'ipg384.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %824 = load float, ptr %"'ipg385.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %825 = fadd fast float %824, %819
  store float %825, ptr %"'ipg385.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %826 = load float, ptr %"'ipg386.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %827 = fadd fast float %826, %817
  store float %827, ptr %"'ipg386.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %828 = load float, ptr %"'ipg387.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %829 = fadd fast float %828, %731
  store float %829, ptr %"'ipg387.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %830 = load float, ptr %"'ipg388.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %831 = fadd fast float %830, %729
  store float %831, ptr %"'ipg388.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %832 = load float, ptr %"'ipg389.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %833 = fadd fast float %832, %727
  store float %833, ptr %"'ipg389.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %834 = load float, ptr %"'ipg390.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %835 = fadd fast float %834, %689
  store float %835, ptr %"'ipg390.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %836 = load float, ptr %"'ipg391.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %837 = fadd fast float %836, %687
  store float %837, ptr %"'ipg391.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %838 = load float, ptr %"'ipg392.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %839 = fadd fast float %838, %685
  store float %839, ptr %"'ipg392.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %840 = load float, ptr %"'ipg393.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %841 = fadd fast float %840, %599
  store float %841, ptr %"'ipg393.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %842 = load float, ptr %"'ipg394.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %843 = fadd fast float %842, %597
  store float %843, ptr %"'ipg394.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %844 = load float, ptr %"'ipg395.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  %845 = fadd fast float %844, %595
  store float %845, ptr %"'ipg395.i", align 4, !tbaa !5, !alias.scope !10, !noalias !7
  call void @_mlir_memref_to_llvm_free(ptr nonnull %"'mi.i")
  call void @_mlir_memref_to_llvm_free(ptr %26)
  call void @llvm.lifetime.end.p0(ptr nonnull %"'ipa5.i")
  call void @llvm.lifetime.end.p0(ptr nonnull %15)
  call void @llvm.lifetime.end.p0(ptr nonnull %"'ipa4.i")
  call void @llvm.lifetime.end.p0(ptr nonnull %16)
  call void @llvm.lifetime.end.p0(ptr nonnull %17)
  call void @llvm.lifetime.end.p0(ptr nonnull %18)
  tail call void @_mlir_memref_to_llvm_free(ptr %25)
  tail call void @_mlir_memref_to_llvm_free(ptr %19)
  %846 = icmp eq ptr %24, inttoptr (i64 3735928559 to ptr)
  br i1 %846, label %847, label %849

847:                                              ; preds = %14
  %848 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 384)
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(384) %848, ptr noundef nonnull align 1 dereferenceable(384) inttoptr (i64 3735928559 to ptr), i64 384, i1 false)
  br label %849

849:                                              ; preds = %847, %14
  %.pn15 = phi ptr [ %848, %847 ], [ %24, %14 ]
  %.pn14 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } poison, ptr %.pn15, 0
  %.pn12 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %.pn14, ptr %.pn15, 1
  %.pn10 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %.pn12, i64 0, 2
  %.pn8 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %.pn10, i64 4, 3, 0
  %.pn6 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %.pn8, i64 8, 3, 1
  %.pn4 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %.pn6, i64 3, 3, 2
  %.pn2 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %.pn4, i64 24, 4, 0
  %.pn = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %.pn2, i64 3, 4, 1
  %850 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %.pn, i64 1, 4, 2
  ret { ptr, ptr, i64, [3 x i64], [3 x i64] } %850
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
  %15 = load float, ptr %14, align 4, !tbaa !5
  %16 = getelementptr inbounds nuw i8, ptr %1, i64 292
  %17 = load float, ptr %16, align 4, !tbaa !5
  %18 = getelementptr inbounds nuw i8, ptr %1, i64 288
  %19 = load float, ptr %18, align 4, !tbaa !5
  %20 = getelementptr inbounds nuw i8, ptr %1, i64 200
  %21 = load float, ptr %20, align 4, !tbaa !5
  %22 = getelementptr inbounds nuw i8, ptr %1, i64 196
  %23 = load float, ptr %22, align 4, !tbaa !5
  %24 = getelementptr inbounds nuw i8, ptr %1, i64 192
  %25 = load float, ptr %24, align 4, !tbaa !5
  %26 = getelementptr inbounds nuw i8, ptr %1, i64 176
  %27 = load float, ptr %26, align 4, !tbaa !5
  %28 = getelementptr inbounds nuw i8, ptr %1, i64 172
  %29 = load float, ptr %28, align 4, !tbaa !5
  %30 = getelementptr inbounds nuw i8, ptr %1, i64 168
  %31 = load float, ptr %30, align 4, !tbaa !5
  %32 = getelementptr inbounds nuw i8, ptr %1, i64 92
  %33 = load float, ptr %32, align 4, !tbaa !5
  %34 = getelementptr inbounds nuw i8, ptr %1, i64 88
  %35 = load float, ptr %34, align 4, !tbaa !5
  %36 = getelementptr inbounds nuw i8, ptr %1, i64 84
  %37 = load float, ptr %36, align 4, !tbaa !5
  %38 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 96)
  %39 = ptrtoint ptr %38 to i64
  %40 = add i64 %39, 63
  %41 = and i64 %40, -64
  %42 = inttoptr i64 %41 to ptr
  store float 0x400921FB60000000, ptr %42, align 64, !tbaa !5
  %43 = getelementptr inbounds nuw i8, ptr %42, i64 4
  store float 0x400921FB60000000, ptr %43, align 4, !tbaa !5
  %44 = getelementptr inbounds nuw i8, ptr %42, i64 8
  store float 0x400921FB60000000, ptr %44, align 8, !tbaa !5
  %45 = getelementptr inbounds nuw i8, ptr %42, i64 12
  store float 0x400921FB60000000, ptr %45, align 4, !tbaa !5
  %46 = getelementptr inbounds nuw i8, ptr %42, i64 16
  store float 0x400921FB60000000, ptr %46, align 16, !tbaa !5
  %47 = getelementptr inbounds nuw i8, ptr %42, i64 20
  store float 0x400921FB60000000, ptr %47, align 4, !tbaa !5
  %48 = getelementptr inbounds nuw i8, ptr %42, i64 24
  store float 0x400921FB60000000, ptr %48, align 8, !tbaa !5
  %49 = getelementptr inbounds nuw i8, ptr %42, i64 28
  store float 0x400921FB60000000, ptr %49, align 4, !tbaa !5
  %50 = load float, ptr %10, align 4, !tbaa !5
  %51 = fmul float %50, 0x400921FB60000000
  store float %51, ptr %42, align 64, !tbaa !5
  %52 = getelementptr inbounds nuw i8, ptr %10, i64 4
  %53 = load float, ptr %52, align 4, !tbaa !5
  %54 = fmul float %53, 0x400921FB60000000
  store float %54, ptr %43, align 4, !tbaa !5
  %55 = getelementptr inbounds nuw i8, ptr %10, i64 8
  %56 = load float, ptr %55, align 4, !tbaa !5
  %57 = fmul float %56, 0x400921FB60000000
  store float %57, ptr %44, align 8, !tbaa !5
  %58 = getelementptr inbounds nuw i8, ptr %10, i64 12
  %59 = load float, ptr %58, align 4, !tbaa !5
  %60 = fmul float %59, 0x400921FB60000000
  store float %60, ptr %45, align 4, !tbaa !5
  %61 = getelementptr inbounds nuw i8, ptr %10, i64 16
  %62 = load float, ptr %61, align 4, !tbaa !5
  %63 = fmul float %62, 0x400921FB60000000
  store float %63, ptr %46, align 16, !tbaa !5
  %64 = getelementptr inbounds nuw i8, ptr %10, i64 20
  %65 = load float, ptr %64, align 4, !tbaa !5
  %66 = fmul float %65, 0x400921FB60000000
  store float %66, ptr %47, align 4, !tbaa !5
  %67 = getelementptr inbounds nuw i8, ptr %10, i64 24
  %68 = load float, ptr %67, align 4, !tbaa !5
  %69 = fmul float %68, 0x400921FB60000000
  store float %69, ptr %48, align 8, !tbaa !5
  %70 = getelementptr inbounds nuw i8, ptr %10, i64 28
  %71 = load float, ptr %70, align 4, !tbaa !5
  %72 = fmul float %71, 0x400921FB60000000
  store float %72, ptr %49, align 4, !tbaa !5
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
  %81 = load float, ptr %80, align 4, !tbaa !5
  %82 = getelementptr inbounds nuw i8, ptr %1, i64 76
  %83 = load float, ptr %82, align 4, !tbaa !5
  %84 = getelementptr inbounds nuw i8, ptr %1, i64 72
  %85 = load float, ptr %84, align 4, !tbaa !5
  %86 = load float, ptr %48, align 8, !tbaa !5
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
  %94 = load float, ptr %93, align 4, !tbaa !5
  %95 = getelementptr inbounds nuw i8, ptr %1, i64 64
  %96 = load float, ptr %95, align 4, !tbaa !5
  %97 = getelementptr inbounds nuw i8, ptr %1, i64 60
  %98 = load float, ptr %97, align 4, !tbaa !5
  %99 = load float, ptr %47, align 4, !tbaa !5
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
  %107 = load float, ptr %106, align 4, !tbaa !5
  %108 = getelementptr inbounds nuw i8, ptr %1, i64 52
  %109 = load float, ptr %108, align 4, !tbaa !5
  %110 = getelementptr inbounds nuw i8, ptr %1, i64 48
  %111 = load float, ptr %110, align 4, !tbaa !5
  %112 = load float, ptr %46, align 16, !tbaa !5
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
  %120 = load float, ptr %119, align 4, !tbaa !5
  %121 = getelementptr inbounds nuw i8, ptr %1, i64 40
  %122 = load float, ptr %121, align 4, !tbaa !5
  %123 = getelementptr inbounds nuw i8, ptr %1, i64 36
  %124 = load float, ptr %123, align 4, !tbaa !5
  %125 = load float, ptr %45, align 4, !tbaa !5
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
  %133 = load float, ptr %132, align 4, !tbaa !5
  %134 = getelementptr inbounds nuw i8, ptr %1, i64 28
  %135 = load float, ptr %134, align 4, !tbaa !5
  %136 = getelementptr inbounds nuw i8, ptr %1, i64 24
  %137 = load float, ptr %136, align 4, !tbaa !5
  %138 = load float, ptr %44, align 8, !tbaa !5
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
  %146 = load float, ptr %145, align 4, !tbaa !5
  %147 = getelementptr inbounds nuw i8, ptr %1, i64 4
  %148 = load float, ptr %147, align 4, !tbaa !5
  %149 = load float, ptr %1, align 4, !tbaa !5
  %150 = load float, ptr %42, align 64, !tbaa !5
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
  %158 = load float, ptr %157, align 4, !tbaa !5
  %159 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %160 = load float, ptr %159, align 4, !tbaa !5
  %161 = getelementptr inbounds nuw i8, ptr %1, i64 12
  %162 = load float, ptr %161, align 4, !tbaa !5
  %163 = load float, ptr %43, align 4, !tbaa !5
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
  %174 = load float, ptr %173, align 4, !tbaa !5
  %175 = getelementptr inbounds nuw i8, ptr %1, i64 148
  %176 = load float, ptr %175, align 4, !tbaa !5
  %177 = getelementptr inbounds nuw i8, ptr %1, i64 144
  %178 = load float, ptr %177, align 4, !tbaa !5
  %179 = fpext float %178 to double
  tail call void @__catalyst__qis__RZ(double %179, ptr %114, ptr null)
  %180 = fpext float %176 to double
  tail call void @__catalyst__qis__RY(double %180, ptr %114, ptr null)
  %181 = fpext float %174 to double
  tail call void @__catalyst__qis__RZ(double %181, ptr %114, ptr null)
  %182 = getelementptr inbounds nuw i8, ptr %1, i64 104
  %183 = load float, ptr %182, align 4, !tbaa !5
  %184 = getelementptr inbounds nuw i8, ptr %1, i64 100
  %185 = load float, ptr %184, align 4, !tbaa !5
  %186 = getelementptr inbounds nuw i8, ptr %1, i64 96
  %187 = load float, ptr %186, align 4, !tbaa !5
  tail call void @__catalyst__qis__CNOT(ptr %75, ptr %152, ptr null)
  %188 = fpext float %187 to double
  tail call void @__catalyst__qis__RZ(double %188, ptr %152, ptr null)
  %189 = fpext float %185 to double
  tail call void @__catalyst__qis__RY(double %189, ptr %152, ptr null)
  %190 = fpext float %183 to double
  tail call void @__catalyst__qis__RZ(double %190, ptr %152, ptr null)
  %191 = getelementptr inbounds nuw i8, ptr %1, i64 128
  %192 = load float, ptr %191, align 4, !tbaa !5
  %193 = getelementptr inbounds nuw i8, ptr %1, i64 124
  %194 = load float, ptr %193, align 4, !tbaa !5
  %195 = getelementptr inbounds nuw i8, ptr %1, i64 120
  %196 = load float, ptr %195, align 4, !tbaa !5
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
  %204 = load float, ptr %203, align 4, !tbaa !5
  %205 = getelementptr inbounds nuw i8, ptr %1, i64 232
  %206 = load float, ptr %205, align 4, !tbaa !5
  %207 = getelementptr inbounds nuw i8, ptr %1, i64 228
  %208 = load float, ptr %207, align 4, !tbaa !5
  %209 = getelementptr inbounds nuw i8, ptr %1, i64 164
  %210 = load float, ptr %209, align 4, !tbaa !5
  %211 = getelementptr inbounds nuw i8, ptr %1, i64 160
  %212 = load float, ptr %211, align 4, !tbaa !5
  %213 = getelementptr inbounds nuw i8, ptr %1, i64 156
  %214 = load float, ptr %213, align 4, !tbaa !5
  %215 = fpext float %214 to double
  tail call void @__catalyst__qis__RZ(double %215, ptr %101, ptr null)
  %216 = fpext float %212 to double
  tail call void @__catalyst__qis__RY(double %216, ptr %101, ptr null)
  %217 = fpext float %210 to double
  tail call void @__catalyst__qis__RZ(double %217, ptr %101, ptr null)
  %218 = getelementptr inbounds nuw i8, ptr %1, i64 116
  %219 = load float, ptr %218, align 4, !tbaa !5
  %220 = getelementptr inbounds nuw i8, ptr %1, i64 112
  %221 = load float, ptr %220, align 4, !tbaa !5
  %222 = getelementptr inbounds nuw i8, ptr %1, i64 108
  %223 = load float, ptr %222, align 4, !tbaa !5
  %224 = fpext float %223 to double
  tail call void @__catalyst__qis__RZ(double %224, ptr %165, ptr null)
  %225 = fpext float %221 to double
  tail call void @__catalyst__qis__RY(double %225, ptr %165, ptr null)
  %226 = fpext float %219 to double
  tail call void @__catalyst__qis__RZ(double %226, ptr %165, ptr null)
  %227 = getelementptr inbounds nuw i8, ptr %1, i64 140
  %228 = load float, ptr %227, align 4, !tbaa !5
  %229 = getelementptr inbounds nuw i8, ptr %1, i64 136
  %230 = load float, ptr %229, align 4, !tbaa !5
  %231 = getelementptr inbounds nuw i8, ptr %1, i64 132
  %232 = load float, ptr %231, align 4, !tbaa !5
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
  %240 = load float, ptr %239, align 4, !tbaa !5
  %241 = getelementptr inbounds nuw i8, ptr %1, i64 220
  %242 = load float, ptr %241, align 4, !tbaa !5
  %243 = getelementptr inbounds nuw i8, ptr %1, i64 216
  %244 = load float, ptr %243, align 4, !tbaa !5
  %245 = fpext float %244 to double
  tail call void @__catalyst__qis__RZ(double %245, ptr %140, ptr null)
  %246 = fpext float %242 to double
  tail call void @__catalyst__qis__RY(double %246, ptr %140, ptr null)
  %247 = fpext float %240 to double
  tail call void @__catalyst__qis__RZ(double %247, ptr %140, ptr null)
  %248 = getelementptr inbounds nuw i8, ptr %1, i64 260
  %249 = load float, ptr %248, align 4, !tbaa !5
  %250 = getelementptr inbounds nuw i8, ptr %1, i64 256
  %251 = load float, ptr %250, align 4, !tbaa !5
  %252 = getelementptr inbounds nuw i8, ptr %1, i64 252
  %253 = load float, ptr %252, align 4, !tbaa !5
  %254 = getelementptr inbounds nuw i8, ptr %1, i64 188
  %255 = load float, ptr %254, align 4, !tbaa !5
  %256 = getelementptr inbounds nuw i8, ptr %1, i64 184
  %257 = load float, ptr %256, align 4, !tbaa !5
  %258 = getelementptr inbounds nuw i8, ptr %1, i64 180
  %259 = load float, ptr %258, align 4, !tbaa !5
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
  %270 = load float, ptr %269, align 4, !tbaa !5
  %271 = getelementptr inbounds nuw i8, ptr %1, i64 340
  %272 = load float, ptr %271, align 4, !tbaa !5
  %273 = getelementptr inbounds nuw i8, ptr %1, i64 336
  %274 = load float, ptr %273, align 4, !tbaa !5
  %275 = getelementptr inbounds nuw i8, ptr %1, i64 284
  %276 = load float, ptr %275, align 4, !tbaa !5
  %277 = getelementptr inbounds nuw i8, ptr %1, i64 280
  %278 = load float, ptr %277, align 4, !tbaa !5
  %279 = getelementptr inbounds nuw i8, ptr %1, i64 276
  %280 = load float, ptr %279, align 4, !tbaa !5
  tail call void @__catalyst__qis__CNOT(ptr %75, ptr %165, ptr null)
  %281 = fpext float %280 to double
  tail call void @__catalyst__qis__RZ(double %281, ptr %75, ptr null)
  %282 = fpext float %278 to double
  tail call void @__catalyst__qis__RY(double %282, ptr %75, ptr null)
  %283 = fpext float %276 to double
  tail call void @__catalyst__qis__RZ(double %283, ptr %75, ptr null)
  %284 = getelementptr inbounds nuw i8, ptr %1, i64 212
  %285 = load float, ptr %284, align 4, !tbaa !5
  %286 = getelementptr inbounds nuw i8, ptr %1, i64 208
  %287 = load float, ptr %286, align 4, !tbaa !5
  %288 = getelementptr inbounds nuw i8, ptr %1, i64 204
  %289 = load float, ptr %288, align 4, !tbaa !5
  %290 = fpext float %289 to double
  tail call void @__catalyst__qis__RZ(double %290, ptr %165, ptr null)
  %291 = fpext float %287 to double
  tail call void @__catalyst__qis__RY(double %291, ptr %165, ptr null)
  %292 = fpext float %285 to double
  tail call void @__catalyst__qis__RZ(double %292, ptr %165, ptr null)
  %293 = getelementptr inbounds nuw i8, ptr %1, i64 248
  %294 = load float, ptr %293, align 4, !tbaa !5
  %295 = getelementptr inbounds nuw i8, ptr %1, i64 244
  %296 = load float, ptr %295, align 4, !tbaa !5
  %297 = getelementptr inbounds nuw i8, ptr %1, i64 240
  %298 = load float, ptr %297, align 4, !tbaa !5
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
  %306 = load float, ptr %305, align 4, !tbaa !5
  %307 = getelementptr inbounds nuw i8, ptr %1, i64 304
  %308 = load float, ptr %307, align 4, !tbaa !5
  %309 = getelementptr inbounds nuw i8, ptr %1, i64 300
  %310 = load float, ptr %309, align 4, !tbaa !5
  %311 = getelementptr inbounds nuw i8, ptr %1, i64 272
  %312 = load float, ptr %311, align 4, !tbaa !5
  %313 = getelementptr inbounds nuw i8, ptr %1, i64 268
  %314 = load float, ptr %313, align 4, !tbaa !5
  %315 = getelementptr inbounds nuw i8, ptr %1, i64 264
  %316 = load float, ptr %315, align 4, !tbaa !5
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
  %324 = load float, ptr %323, align 4, !tbaa !5
  %325 = getelementptr inbounds nuw i8, ptr %1, i64 352
  %326 = load float, ptr %325, align 4, !tbaa !5
  %327 = getelementptr inbounds nuw i8, ptr %1, i64 348
  %328 = load float, ptr %327, align 4, !tbaa !5
  %329 = fpext float %328 to double
  tail call void @__catalyst__qis__RZ(double %329, ptr %101, ptr null)
  %330 = fpext float %326 to double
  tail call void @__catalyst__qis__RY(double %330, ptr %101, ptr null)
  %331 = fpext float %324 to double
  tail call void @__catalyst__qis__RZ(double %331, ptr %101, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %165, ptr %101, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %101, ptr %165, ptr null)
  %332 = getelementptr inbounds nuw i8, ptr %1, i64 320
  %333 = load float, ptr %332, align 4, !tbaa !5
  %334 = getelementptr inbounds nuw i8, ptr %1, i64 316
  %335 = load float, ptr %334, align 4, !tbaa !5
  %336 = getelementptr inbounds nuw i8, ptr %1, i64 312
  %337 = load float, ptr %336, align 4, !tbaa !5
  tail call void @__catalyst__qis__CNOT(ptr %75, ptr %140, ptr null)
  %338 = fpext float %337 to double
  tail call void @__catalyst__qis__RZ(double %338, ptr %140, ptr null)
  %339 = fpext float %335 to double
  tail call void @__catalyst__qis__RY(double %339, ptr %140, ptr null)
  %340 = fpext float %333 to double
  tail call void @__catalyst__qis__RZ(double %340, ptr %140, ptr null)
  %341 = getelementptr inbounds nuw i8, ptr %1, i64 368
  %342 = load float, ptr %341, align 4, !tbaa !5
  %343 = getelementptr inbounds nuw i8, ptr %1, i64 364
  %344 = load float, ptr %343, align 4, !tbaa !5
  %345 = getelementptr inbounds nuw i8, ptr %1, i64 360
  %346 = load float, ptr %345, align 4, !tbaa !5
  %347 = fpext float %346 to double
  tail call void @__catalyst__qis__RZ(double %347, ptr %88, ptr null)
  %348 = fpext float %344 to double
  tail call void @__catalyst__qis__RY(double %348, ptr %88, ptr null)
  %349 = fpext float %342 to double
  tail call void @__catalyst__qis__RZ(double %349, ptr %88, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %140, ptr %88, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %88, ptr %140, ptr null)
  %350 = getelementptr inbounds nuw i8, ptr %1, i64 332
  %351 = load float, ptr %350, align 4, !tbaa !5
  %352 = getelementptr inbounds nuw i8, ptr %1, i64 328
  %353 = load float, ptr %352, align 4, !tbaa !5
  %354 = getelementptr inbounds nuw i8, ptr %1, i64 324
  %355 = load float, ptr %354, align 4, !tbaa !5
  %356 = fpext float %355 to double
  tail call void @__catalyst__qis__RZ(double %356, ptr %127, ptr null)
  %357 = fpext float %353 to double
  tail call void @__catalyst__qis__RY(double %357, ptr %127, ptr null)
  %358 = fpext float %351 to double
  tail call void @__catalyst__qis__RZ(double %358, ptr %127, ptr null)
  %359 = getelementptr inbounds nuw i8, ptr %1, i64 380
  %360 = load float, ptr %359, align 4, !tbaa !5
  %361 = getelementptr inbounds nuw i8, ptr %1, i64 376
  %362 = load float, ptr %361, align 4, !tbaa !5
  %363 = getelementptr inbounds nuw i8, ptr %1, i64 372
  %364 = load float, ptr %363, align 4, !tbaa !5
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
define noundef i64 @qnode_forward_0.pcount(ptr readnone captures(none) %0, ptr readnone captures(none) %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, ptr readnone captures(none) %9, ptr readnone captures(none) %10, i64 %11, i64 %12, i64 %13) local_unnamed_addr #2 {
  ret i64 104
}

define void @qnode_forward_0.quantum.customqgrad(ptr readonly captures(none) %0, ptr readnone captures(none) %1, ptr readonly captures(none) %2, ptr readnone captures(none) %3, ptr readnone captures(none) %4, ptr readonly captures(none) %5, ptr readnone captures(none) %6, ptr readonly captures(none) %7, ptr readnone captures(none) %8) local_unnamed_addr #3 {
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
  %16 = load double, ptr %.unpack45, align 8, !tbaa !2
  %17 = getelementptr inbounds nuw double, ptr %11, i64 %15
  %18 = load double, ptr %17, align 8, !tbaa !2
  %19 = getelementptr inbounds nuw double, ptr %.unpack34, i64 %15
  %20 = load double, ptr %19, align 8, !tbaa !2
  %21 = fmul double %16, %18
  %22 = fadd double %20, %21
  store double %22, ptr %19, align 8, !tbaa !2
  %23 = add nuw nsw i64 %15, 1
  %exitcond.not = icmp eq i64 %23, %.unpack38.unpack
  br i1 %exitcond.not, label %._crit_edge, label %.lr.ph

._crit_edge:                                      ; preds = %.lr.ph, %9
  ret void
}

; Function Attrs: noinline
define void @qnode_forward_0.quantum(ptr %0, ptr %1, ptr %2, ptr %3) local_unnamed_addr #4 !enzyme_augment !52 !enzyme_gradient !53 {
  %5 = load volatile { ptr, ptr, i64, [3 x i64], [3 x i64] }, ptr %0, align 8
  %6 = load volatile { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %1, align 8
  %7 = load volatile { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %2, align 8
  %8 = load volatile { ptr, ptr, i64 }, ptr %3, align 8
  tail call void @__catalyst__rt__device_init(ptr nonnull @"/home/pichau/QAgents/.venv/lib/python3.12/site-packages/pennylane_lightning/liblightning_gpu_catalyst.so", ptr nonnull @LightningGPUSimulator, ptr nonnull @"{}", i64 0, i1 false)
  %9 = tail call ptr @__catalyst__rt__qubit_allocate_array(i64 8)
  %10 = tail call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %9, i64 7)
  %11 = load ptr, ptr %10, align 8
  %12 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %13 = load double, ptr %12, align 8, !tbaa !2
  tail call void @__catalyst__qis__RY(double %13, ptr %11, ptr null)
  %14 = getelementptr inbounds nuw i8, ptr %12, i64 8
  %15 = load double, ptr %14, align 8, !tbaa !2
  tail call void @__catalyst__qis__RZ(double %15, ptr %11, ptr null)
  %16 = getelementptr inbounds nuw i8, ptr %12, i64 16
  %17 = load double, ptr %16, align 8, !tbaa !2
  tail call void @__catalyst__qis__RY(double %17, ptr %11, ptr null)
  %18 = getelementptr inbounds nuw i8, ptr %12, i64 24
  %19 = load double, ptr %18, align 8, !tbaa !2
  tail call void @__catalyst__qis__RZ(double %19, ptr %11, ptr null)
  %20 = tail call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %9, i64 6)
  %21 = load ptr, ptr %20, align 8
  %22 = getelementptr inbounds nuw i8, ptr %12, i64 32
  %23 = load double, ptr %22, align 8, !tbaa !2
  tail call void @__catalyst__qis__RY(double %23, ptr %21, ptr null)
  %24 = getelementptr inbounds nuw i8, ptr %12, i64 40
  %25 = load double, ptr %24, align 8, !tbaa !2
  tail call void @__catalyst__qis__RZ(double %25, ptr %21, ptr null)
  %26 = getelementptr inbounds nuw i8, ptr %12, i64 48
  %27 = load double, ptr %26, align 8, !tbaa !2
  tail call void @__catalyst__qis__RY(double %27, ptr %21, ptr null)
  %28 = getelementptr inbounds nuw i8, ptr %12, i64 56
  %29 = load double, ptr %28, align 8, !tbaa !2
  tail call void @__catalyst__qis__RZ(double %29, ptr %21, ptr null)
  %30 = tail call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %9, i64 5)
  %31 = load ptr, ptr %30, align 8
  %32 = getelementptr inbounds nuw i8, ptr %12, i64 64
  %33 = load double, ptr %32, align 8, !tbaa !2
  tail call void @__catalyst__qis__RY(double %33, ptr %31, ptr null)
  %34 = getelementptr inbounds nuw i8, ptr %12, i64 72
  %35 = load double, ptr %34, align 8, !tbaa !2
  tail call void @__catalyst__qis__RZ(double %35, ptr %31, ptr null)
  %36 = getelementptr inbounds nuw i8, ptr %12, i64 80
  %37 = load double, ptr %36, align 8, !tbaa !2
  tail call void @__catalyst__qis__RY(double %37, ptr %31, ptr null)
  %38 = getelementptr inbounds nuw i8, ptr %12, i64 88
  %39 = load double, ptr %38, align 8, !tbaa !2
  tail call void @__catalyst__qis__RZ(double %39, ptr %31, ptr null)
  %40 = tail call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %9, i64 4)
  %41 = load ptr, ptr %40, align 8
  %42 = getelementptr inbounds nuw i8, ptr %12, i64 96
  %43 = load double, ptr %42, align 8, !tbaa !2
  tail call void @__catalyst__qis__RY(double %43, ptr %41, ptr null)
  %44 = getelementptr inbounds nuw i8, ptr %12, i64 104
  %45 = load double, ptr %44, align 8, !tbaa !2
  tail call void @__catalyst__qis__RZ(double %45, ptr %41, ptr null)
  %46 = getelementptr inbounds nuw i8, ptr %12, i64 112
  %47 = load double, ptr %46, align 8, !tbaa !2
  tail call void @__catalyst__qis__RY(double %47, ptr %41, ptr null)
  %48 = getelementptr inbounds nuw i8, ptr %12, i64 120
  %49 = load double, ptr %48, align 8, !tbaa !2
  tail call void @__catalyst__qis__RZ(double %49, ptr %41, ptr null)
  %50 = tail call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %9, i64 3)
  %51 = load ptr, ptr %50, align 8
  %52 = getelementptr inbounds nuw i8, ptr %12, i64 128
  %53 = load double, ptr %52, align 8, !tbaa !2
  tail call void @__catalyst__qis__RY(double %53, ptr %51, ptr null)
  %54 = getelementptr inbounds nuw i8, ptr %12, i64 136
  %55 = load double, ptr %54, align 8, !tbaa !2
  tail call void @__catalyst__qis__RZ(double %55, ptr %51, ptr null)
  %56 = getelementptr inbounds nuw i8, ptr %12, i64 144
  %57 = load double, ptr %56, align 8, !tbaa !2
  tail call void @__catalyst__qis__RY(double %57, ptr %51, ptr null)
  %58 = getelementptr inbounds nuw i8, ptr %12, i64 152
  %59 = load double, ptr %58, align 8, !tbaa !2
  tail call void @__catalyst__qis__RZ(double %59, ptr %51, ptr null)
  %60 = tail call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %9, i64 2)
  %61 = load ptr, ptr %60, align 8
  %62 = getelementptr inbounds nuw i8, ptr %12, i64 160
  %63 = load double, ptr %62, align 8, !tbaa !2
  tail call void @__catalyst__qis__RY(double %63, ptr %61, ptr null)
  %64 = getelementptr inbounds nuw i8, ptr %12, i64 168
  %65 = load double, ptr %64, align 8, !tbaa !2
  tail call void @__catalyst__qis__RZ(double %65, ptr %61, ptr null)
  %66 = getelementptr inbounds nuw i8, ptr %12, i64 176
  %67 = load double, ptr %66, align 8, !tbaa !2
  tail call void @__catalyst__qis__RY(double %67, ptr %61, ptr null)
  %68 = getelementptr inbounds nuw i8, ptr %12, i64 184
  %69 = load double, ptr %68, align 8, !tbaa !2
  tail call void @__catalyst__qis__RZ(double %69, ptr %61, ptr null)
  %70 = tail call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %9, i64 0)
  %71 = load ptr, ptr %70, align 8
  %72 = getelementptr inbounds nuw i8, ptr %12, i64 192
  %73 = load double, ptr %72, align 8, !tbaa !2
  tail call void @__catalyst__qis__RY(double %73, ptr %71, ptr null)
  %74 = getelementptr inbounds nuw i8, ptr %12, i64 200
  %75 = load double, ptr %74, align 8, !tbaa !2
  tail call void @__catalyst__qis__RZ(double %75, ptr %71, ptr null)
  %76 = getelementptr inbounds nuw i8, ptr %12, i64 208
  %77 = load double, ptr %76, align 8, !tbaa !2
  tail call void @__catalyst__qis__RY(double %77, ptr %71, ptr null)
  %78 = getelementptr inbounds nuw i8, ptr %12, i64 216
  %79 = load double, ptr %78, align 8, !tbaa !2
  tail call void @__catalyst__qis__RZ(double %79, ptr %71, ptr null)
  %80 = tail call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %9, i64 1)
  %81 = load ptr, ptr %80, align 8
  %82 = getelementptr inbounds nuw i8, ptr %12, i64 224
  %83 = load double, ptr %82, align 8, !tbaa !2
  tail call void @__catalyst__qis__RY(double %83, ptr %81, ptr null)
  %84 = getelementptr inbounds nuw i8, ptr %12, i64 232
  %85 = load double, ptr %84, align 8, !tbaa !2
  tail call void @__catalyst__qis__RZ(double %85, ptr %81, ptr null)
  %86 = getelementptr inbounds nuw i8, ptr %12, i64 240
  %87 = load double, ptr %86, align 8, !tbaa !2
  tail call void @__catalyst__qis__RY(double %87, ptr %81, ptr null)
  %88 = getelementptr inbounds nuw i8, ptr %12, i64 248
  %89 = load double, ptr %88, align 8, !tbaa !2
  tail call void @__catalyst__qis__RZ(double %89, ptr %81, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %71, ptr %81, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %81, ptr %61, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %61, ptr %51, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %51, ptr %41, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %41, ptr %31, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %31, ptr %21, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %21, ptr %11, ptr null)
  %90 = getelementptr inbounds nuw i8, ptr %12, i64 256
  %91 = load double, ptr %90, align 8, !tbaa !2
  tail call void @__catalyst__qis__RZ(double %91, ptr %21, ptr null)
  %92 = getelementptr inbounds nuw i8, ptr %12, i64 264
  %93 = load double, ptr %92, align 8, !tbaa !2
  tail call void @__catalyst__qis__RY(double %93, ptr %21, ptr null)
  %94 = getelementptr inbounds nuw i8, ptr %12, i64 272
  %95 = load double, ptr %94, align 8, !tbaa !2
  tail call void @__catalyst__qis__RZ(double %95, ptr %21, ptr null)
  %96 = getelementptr inbounds nuw i8, ptr %12, i64 280
  %97 = load double, ptr %96, align 8, !tbaa !2
  tail call void @__catalyst__qis__RZ(double %97, ptr %41, ptr null)
  %98 = getelementptr inbounds nuw i8, ptr %12, i64 288
  %99 = load double, ptr %98, align 8, !tbaa !2
  tail call void @__catalyst__qis__RY(double %99, ptr %41, ptr null)
  %100 = getelementptr inbounds nuw i8, ptr %12, i64 296
  %101 = load double, ptr %100, align 8, !tbaa !2
  tail call void @__catalyst__qis__RZ(double %101, ptr %41, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %11, ptr %71, ptr null)
  %102 = getelementptr inbounds nuw i8, ptr %12, i64 304
  %103 = load double, ptr %102, align 8, !tbaa !2
  tail call void @__catalyst__qis__RZ(double %103, ptr %71, ptr null)
  %104 = getelementptr inbounds nuw i8, ptr %12, i64 312
  %105 = load double, ptr %104, align 8, !tbaa !2
  tail call void @__catalyst__qis__RY(double %105, ptr %71, ptr null)
  %106 = getelementptr inbounds nuw i8, ptr %12, i64 320
  %107 = load double, ptr %106, align 8, !tbaa !2
  tail call void @__catalyst__qis__RZ(double %107, ptr %71, ptr null)
  %108 = getelementptr inbounds nuw i8, ptr %12, i64 328
  %109 = load double, ptr %108, align 8, !tbaa !2
  tail call void @__catalyst__qis__RZ(double %109, ptr %61, ptr null)
  %110 = getelementptr inbounds nuw i8, ptr %12, i64 336
  %111 = load double, ptr %110, align 8, !tbaa !2
  tail call void @__catalyst__qis__RY(double %111, ptr %61, ptr null)
  %112 = getelementptr inbounds nuw i8, ptr %12, i64 344
  %113 = load double, ptr %112, align 8, !tbaa !2
  tail call void @__catalyst__qis__RZ(double %113, ptr %61, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %71, ptr %61, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %61, ptr %41, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %41, ptr %21, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %21, ptr %71, ptr null)
  %114 = getelementptr inbounds nuw i8, ptr %12, i64 352
  %115 = load double, ptr %114, align 8, !tbaa !2
  tail call void @__catalyst__qis__RZ(double %115, ptr %71, ptr null)
  %116 = getelementptr inbounds nuw i8, ptr %12, i64 360
  %117 = load double, ptr %116, align 8, !tbaa !2
  tail call void @__catalyst__qis__RY(double %117, ptr %71, ptr null)
  %118 = getelementptr inbounds nuw i8, ptr %12, i64 368
  %119 = load double, ptr %118, align 8, !tbaa !2
  tail call void @__catalyst__qis__RZ(double %119, ptr %71, ptr null)
  %120 = getelementptr inbounds nuw i8, ptr %12, i64 376
  %121 = load double, ptr %120, align 8, !tbaa !2
  tail call void @__catalyst__qis__RZ(double %121, ptr %31, ptr null)
  %122 = getelementptr inbounds nuw i8, ptr %12, i64 384
  %123 = load double, ptr %122, align 8, !tbaa !2
  tail call void @__catalyst__qis__RY(double %123, ptr %31, ptr null)
  %124 = getelementptr inbounds nuw i8, ptr %12, i64 392
  %125 = load double, ptr %124, align 8, !tbaa !2
  tail call void @__catalyst__qis__RZ(double %125, ptr %31, ptr null)
  %126 = getelementptr inbounds nuw i8, ptr %12, i64 400
  %127 = load double, ptr %126, align 8, !tbaa !2
  tail call void @__catalyst__qis__RZ(double %127, ptr %81, ptr null)
  %128 = getelementptr inbounds nuw i8, ptr %12, i64 408
  %129 = load double, ptr %128, align 8, !tbaa !2
  tail call void @__catalyst__qis__RY(double %129, ptr %81, ptr null)
  %130 = getelementptr inbounds nuw i8, ptr %12, i64 416
  %131 = load double, ptr %130, align 8, !tbaa !2
  tail call void @__catalyst__qis__RZ(double %131, ptr %81, ptr null)
  %132 = getelementptr inbounds nuw i8, ptr %12, i64 424
  %133 = load double, ptr %132, align 8, !tbaa !2
  tail call void @__catalyst__qis__RZ(double %133, ptr %51, ptr null)
  %134 = getelementptr inbounds nuw i8, ptr %12, i64 432
  %135 = load double, ptr %134, align 8, !tbaa !2
  tail call void @__catalyst__qis__RY(double %135, ptr %51, ptr null)
  %136 = getelementptr inbounds nuw i8, ptr %12, i64 440
  %137 = load double, ptr %136, align 8, !tbaa !2
  tail call void @__catalyst__qis__RZ(double %137, ptr %51, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %81, ptr %51, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %51, ptr %31, ptr null)
  %138 = getelementptr inbounds nuw i8, ptr %12, i64 448
  %139 = load double, ptr %138, align 8, !tbaa !2
  tail call void @__catalyst__qis__RZ(double %139, ptr %51, ptr null)
  %140 = getelementptr inbounds nuw i8, ptr %12, i64 456
  %141 = load double, ptr %140, align 8, !tbaa !2
  tail call void @__catalyst__qis__RY(double %141, ptr %51, ptr null)
  %142 = getelementptr inbounds nuw i8, ptr %12, i64 464
  %143 = load double, ptr %142, align 8, !tbaa !2
  tail call void @__catalyst__qis__RZ(double %143, ptr %51, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %71, ptr %51, ptr null)
  %144 = getelementptr inbounds nuw i8, ptr %12, i64 472
  %145 = load double, ptr %144, align 8, !tbaa !2
  tail call void @__catalyst__qis__RZ(double %145, ptr %61, ptr null)
  %146 = getelementptr inbounds nuw i8, ptr %12, i64 480
  %147 = load double, ptr %146, align 8, !tbaa !2
  tail call void @__catalyst__qis__RY(double %147, ptr %61, ptr null)
  %148 = getelementptr inbounds nuw i8, ptr %12, i64 488
  %149 = load double, ptr %148, align 8, !tbaa !2
  tail call void @__catalyst__qis__RZ(double %149, ptr %61, ptr null)
  %150 = getelementptr inbounds nuw i8, ptr %12, i64 496
  %151 = load double, ptr %150, align 8, !tbaa !2
  tail call void @__catalyst__qis__RZ(double %151, ptr %11, ptr null)
  %152 = getelementptr inbounds nuw i8, ptr %12, i64 504
  %153 = load double, ptr %152, align 8, !tbaa !2
  tail call void @__catalyst__qis__RY(double %153, ptr %11, ptr null)
  %154 = getelementptr inbounds nuw i8, ptr %12, i64 512
  %155 = load double, ptr %154, align 8, !tbaa !2
  tail call void @__catalyst__qis__RZ(double %155, ptr %11, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %31, ptr %11, ptr null)
  %156 = getelementptr inbounds nuw i8, ptr %12, i64 520
  %157 = load double, ptr %156, align 8, !tbaa !2
  tail call void @__catalyst__qis__RZ(double %157, ptr %31, ptr null)
  %158 = getelementptr inbounds nuw i8, ptr %12, i64 528
  %159 = load double, ptr %158, align 8, !tbaa !2
  tail call void @__catalyst__qis__RY(double %159, ptr %31, ptr null)
  %160 = getelementptr inbounds nuw i8, ptr %12, i64 536
  %161 = load double, ptr %160, align 8, !tbaa !2
  tail call void @__catalyst__qis__RZ(double %161, ptr %31, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %61, ptr %31, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %31, ptr %71, ptr null)
  %162 = getelementptr inbounds nuw i8, ptr %12, i64 544
  %163 = load double, ptr %162, align 8, !tbaa !2
  tail call void @__catalyst__qis__RZ(double %163, ptr %71, ptr null)
  %164 = getelementptr inbounds nuw i8, ptr %12, i64 552
  %165 = load double, ptr %164, align 8, !tbaa !2
  tail call void @__catalyst__qis__RY(double %165, ptr %71, ptr null)
  %166 = getelementptr inbounds nuw i8, ptr %12, i64 560
  %167 = load double, ptr %166, align 8, !tbaa !2
  tail call void @__catalyst__qis__RZ(double %167, ptr %71, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %11, ptr %81, ptr null)
  %168 = getelementptr inbounds nuw i8, ptr %12, i64 568
  %169 = load double, ptr %168, align 8, !tbaa !2
  tail call void @__catalyst__qis__RZ(double %169, ptr %11, ptr null)
  %170 = getelementptr inbounds nuw i8, ptr %12, i64 576
  %171 = load double, ptr %170, align 8, !tbaa !2
  tail call void @__catalyst__qis__RY(double %171, ptr %11, ptr null)
  %172 = getelementptr inbounds nuw i8, ptr %12, i64 584
  %173 = load double, ptr %172, align 8, !tbaa !2
  tail call void @__catalyst__qis__RZ(double %173, ptr %11, ptr null)
  %174 = getelementptr inbounds nuw i8, ptr %12, i64 592
  %175 = load double, ptr %174, align 8, !tbaa !2
  tail call void @__catalyst__qis__RZ(double %175, ptr %81, ptr null)
  %176 = getelementptr inbounds nuw i8, ptr %12, i64 600
  %177 = load double, ptr %176, align 8, !tbaa !2
  tail call void @__catalyst__qis__RY(double %177, ptr %81, ptr null)
  %178 = getelementptr inbounds nuw i8, ptr %12, i64 608
  %179 = load double, ptr %178, align 8, !tbaa !2
  tail call void @__catalyst__qis__RZ(double %179, ptr %81, ptr null)
  %180 = getelementptr inbounds nuw i8, ptr %12, i64 616
  %181 = load double, ptr %180, align 8, !tbaa !2
  tail call void @__catalyst__qis__RZ(double %181, ptr %41, ptr null)
  %182 = getelementptr inbounds nuw i8, ptr %12, i64 624
  %183 = load double, ptr %182, align 8, !tbaa !2
  tail call void @__catalyst__qis__RY(double %183, ptr %41, ptr null)
  %184 = getelementptr inbounds nuw i8, ptr %12, i64 632
  %185 = load double, ptr %184, align 8, !tbaa !2
  tail call void @__catalyst__qis__RZ(double %185, ptr %41, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %81, ptr %41, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %41, ptr %11, ptr null)
  %186 = getelementptr inbounds nuw i8, ptr %12, i64 640
  %187 = load double, ptr %186, align 8, !tbaa !2
  tail call void @__catalyst__qis__RZ(double %187, ptr %41, ptr null)
  %188 = getelementptr inbounds nuw i8, ptr %12, i64 648
  %189 = load double, ptr %188, align 8, !tbaa !2
  tail call void @__catalyst__qis__RY(double %189, ptr %41, ptr null)
  %190 = getelementptr inbounds nuw i8, ptr %12, i64 656
  %191 = load double, ptr %190, align 8, !tbaa !2
  tail call void @__catalyst__qis__RZ(double %191, ptr %41, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %71, ptr %41, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %41, ptr %71, ptr null)
  %192 = getelementptr inbounds nuw i8, ptr %12, i64 664
  %193 = load double, ptr %192, align 8, !tbaa !2
  tail call void @__catalyst__qis__RZ(double %193, ptr %21, ptr null)
  %194 = getelementptr inbounds nuw i8, ptr %12, i64 672
  %195 = load double, ptr %194, align 8, !tbaa !2
  tail call void @__catalyst__qis__RY(double %195, ptr %21, ptr null)
  %196 = getelementptr inbounds nuw i8, ptr %12, i64 680
  %197 = load double, ptr %196, align 8, !tbaa !2
  tail call void @__catalyst__qis__RZ(double %197, ptr %21, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %51, ptr %21, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %21, ptr %81, ptr null)
  %198 = getelementptr inbounds nuw i8, ptr %12, i64 688
  %199 = load double, ptr %198, align 8, !tbaa !2
  tail call void @__catalyst__qis__RZ(double %199, ptr %81, ptr null)
  %200 = getelementptr inbounds nuw i8, ptr %12, i64 696
  %201 = load double, ptr %200, align 8, !tbaa !2
  tail call void @__catalyst__qis__RY(double %201, ptr %81, ptr null)
  %202 = getelementptr inbounds nuw i8, ptr %12, i64 704
  %203 = load double, ptr %202, align 8, !tbaa !2
  tail call void @__catalyst__qis__RZ(double %203, ptr %81, ptr null)
  %204 = getelementptr inbounds nuw i8, ptr %12, i64 712
  %205 = load double, ptr %204, align 8, !tbaa !2
  tail call void @__catalyst__qis__RZ(double %205, ptr %31, ptr null)
  %206 = getelementptr inbounds nuw i8, ptr %12, i64 720
  %207 = load double, ptr %206, align 8, !tbaa !2
  tail call void @__catalyst__qis__RY(double %207, ptr %31, ptr null)
  %208 = getelementptr inbounds nuw i8, ptr %12, i64 728
  %209 = load double, ptr %208, align 8, !tbaa !2
  tail call void @__catalyst__qis__RZ(double %209, ptr %31, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %81, ptr %31, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %31, ptr %81, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %11, ptr %61, ptr null)
  %210 = getelementptr inbounds nuw i8, ptr %12, i64 736
  %211 = load double, ptr %210, align 8, !tbaa !2
  tail call void @__catalyst__qis__RZ(double %211, ptr %61, ptr null)
  %212 = getelementptr inbounds nuw i8, ptr %12, i64 744
  %213 = load double, ptr %212, align 8, !tbaa !2
  tail call void @__catalyst__qis__RY(double %213, ptr %61, ptr null)
  %214 = getelementptr inbounds nuw i8, ptr %12, i64 752
  %215 = load double, ptr %214, align 8, !tbaa !2
  tail call void @__catalyst__qis__RZ(double %215, ptr %61, ptr null)
  %216 = getelementptr inbounds nuw i8, ptr %12, i64 760
  %217 = load double, ptr %216, align 8, !tbaa !2
  tail call void @__catalyst__qis__RZ(double %217, ptr %21, ptr null)
  %218 = getelementptr inbounds nuw i8, ptr %12, i64 768
  %219 = load double, ptr %218, align 8, !tbaa !2
  tail call void @__catalyst__qis__RY(double %219, ptr %21, ptr null)
  %220 = getelementptr inbounds nuw i8, ptr %12, i64 776
  %221 = load double, ptr %220, align 8, !tbaa !2
  tail call void @__catalyst__qis__RZ(double %221, ptr %21, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %61, ptr %21, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %21, ptr %61, ptr null)
  %222 = getelementptr inbounds nuw i8, ptr %12, i64 784
  %223 = load double, ptr %222, align 8, !tbaa !2
  tail call void @__catalyst__qis__RZ(double %223, ptr %51, ptr null)
  %224 = getelementptr inbounds nuw i8, ptr %12, i64 792
  %225 = load double, ptr %224, align 8, !tbaa !2
  tail call void @__catalyst__qis__RY(double %225, ptr %51, ptr null)
  %226 = getelementptr inbounds nuw i8, ptr %12, i64 800
  %227 = load double, ptr %226, align 8, !tbaa !2
  tail call void @__catalyst__qis__RZ(double %227, ptr %51, ptr null)
  %228 = getelementptr inbounds nuw i8, ptr %12, i64 808
  %229 = load double, ptr %228, align 8, !tbaa !2
  tail call void @__catalyst__qis__RZ(double %229, ptr %11, ptr null)
  %230 = getelementptr inbounds nuw i8, ptr %12, i64 816
  %231 = load double, ptr %230, align 8, !tbaa !2
  tail call void @__catalyst__qis__RY(double %231, ptr %11, ptr null)
  %232 = getelementptr inbounds nuw i8, ptr %12, i64 824
  %233 = load double, ptr %232, align 8, !tbaa !2
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
  store double %235, ptr %240, align 64, !tbaa !2
  tail call void @__catalyst__rt__qubit_release_array(ptr %9)
  tail call void @__catalyst__rt__device_release()
  %241 = load double, ptr %240, align 64, !tbaa !2
  %242 = extractvalue { ptr, ptr, i64 } %8, 1
  store double %241, ptr %242, align 8, !tbaa !2
  ret void
}

define noalias noundef ptr @qnode_forward_0.quantum.augfwd(ptr %0, ptr readnone captures(none) %1, ptr %2, ptr readnone captures(none) %3, ptr %4, ptr readnone captures(none) %5, ptr %6, ptr readnone captures(none) %7) local_unnamed_addr #3 {
  tail call void @qnode_forward_0.quantum(ptr %0, ptr %2, ptr %4, ptr %6)
  ret ptr null
}

define void @qnode_forward_0.preprocess(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, ptr %9, ptr %10, i64 %11, i64 %12, i64 %13, i64 %14, ptr readnone captures(none) %15, ptr writeonly captures(none) initializes((0, 8)) %16, i64 %17) local_unnamed_addr {
.preheader.preheader:
  %18 = alloca { ptr, ptr, i64 }, align 8
  %19 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, align 8
  %20 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, align 8
  %21 = alloca { ptr, ptr, i64, [3 x i64], [3 x i64] }, align 8
  %.idx = shl i64 %14, 3
  %22 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 %.idx)
  %23 = getelementptr inbounds nuw i8, ptr %1, i64 296
  %24 = load float, ptr %23, align 4, !tbaa !5
  %25 = getelementptr inbounds nuw i8, ptr %1, i64 292
  %26 = load float, ptr %25, align 4, !tbaa !5
  %27 = getelementptr inbounds nuw i8, ptr %1, i64 288
  %28 = load float, ptr %27, align 4, !tbaa !5
  %29 = getelementptr inbounds nuw i8, ptr %1, i64 200
  %30 = load float, ptr %29, align 4, !tbaa !5
  %31 = getelementptr inbounds nuw i8, ptr %1, i64 196
  %32 = load float, ptr %31, align 4, !tbaa !5
  %33 = getelementptr inbounds nuw i8, ptr %1, i64 192
  %34 = load float, ptr %33, align 4, !tbaa !5
  %35 = getelementptr inbounds nuw i8, ptr %1, i64 176
  %36 = load float, ptr %35, align 4, !tbaa !5
  %37 = getelementptr inbounds nuw i8, ptr %1, i64 172
  %38 = load float, ptr %37, align 4, !tbaa !5
  %39 = getelementptr inbounds nuw i8, ptr %1, i64 168
  %40 = load float, ptr %39, align 4, !tbaa !5
  %41 = getelementptr inbounds nuw i8, ptr %1, i64 92
  %42 = load float, ptr %41, align 4, !tbaa !5
  %43 = getelementptr inbounds nuw i8, ptr %1, i64 88
  %44 = load float, ptr %43, align 4, !tbaa !5
  %45 = getelementptr inbounds nuw i8, ptr %1, i64 84
  %46 = load float, ptr %45, align 4, !tbaa !5
  %47 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 96)
  %48 = ptrtoint ptr %47 to i64
  %49 = add i64 %48, 63
  %50 = and i64 %49, -64
  %51 = inttoptr i64 %50 to ptr
  store float 0x400921FB60000000, ptr %51, align 64, !tbaa !5
  %52 = getelementptr inbounds nuw i8, ptr %51, i64 4
  store float 0x400921FB60000000, ptr %52, align 4, !tbaa !5
  %53 = getelementptr inbounds nuw i8, ptr %51, i64 8
  store float 0x400921FB60000000, ptr %53, align 8, !tbaa !5
  %54 = getelementptr inbounds nuw i8, ptr %51, i64 12
  store float 0x400921FB60000000, ptr %54, align 4, !tbaa !5
  %55 = getelementptr inbounds nuw i8, ptr %51, i64 16
  store float 0x400921FB60000000, ptr %55, align 16, !tbaa !5
  %56 = getelementptr inbounds nuw i8, ptr %51, i64 20
  store float 0x400921FB60000000, ptr %56, align 4, !tbaa !5
  %57 = getelementptr inbounds nuw i8, ptr %51, i64 24
  store float 0x400921FB60000000, ptr %57, align 8, !tbaa !5
  %58 = getelementptr inbounds nuw i8, ptr %51, i64 28
  store float 0x400921FB60000000, ptr %58, align 4, !tbaa !5
  %59 = load float, ptr %10, align 4, !tbaa !5
  %60 = fmul float %59, 0x400921FB60000000
  store float %60, ptr %51, align 64, !tbaa !5
  %61 = getelementptr inbounds nuw i8, ptr %10, i64 4
  %62 = load float, ptr %61, align 4, !tbaa !5
  %63 = fmul float %62, 0x400921FB60000000
  store float %63, ptr %52, align 4, !tbaa !5
  %64 = getelementptr inbounds nuw i8, ptr %10, i64 8
  %65 = load float, ptr %64, align 4, !tbaa !5
  %66 = fmul float %65, 0x400921FB60000000
  store float %66, ptr %53, align 8, !tbaa !5
  %67 = getelementptr inbounds nuw i8, ptr %10, i64 12
  %68 = load float, ptr %67, align 4, !tbaa !5
  %69 = fmul float %68, 0x400921FB60000000
  store float %69, ptr %54, align 4, !tbaa !5
  %70 = getelementptr inbounds nuw i8, ptr %10, i64 16
  %71 = load float, ptr %70, align 4, !tbaa !5
  %72 = fmul float %71, 0x400921FB60000000
  store float %72, ptr %55, align 16, !tbaa !5
  %73 = getelementptr inbounds nuw i8, ptr %10, i64 20
  %74 = load float, ptr %73, align 4, !tbaa !5
  %75 = fmul float %74, 0x400921FB60000000
  store float %75, ptr %56, align 4, !tbaa !5
  %76 = getelementptr inbounds nuw i8, ptr %10, i64 24
  %77 = load float, ptr %76, align 4, !tbaa !5
  %78 = fmul float %77, 0x400921FB60000000
  store float %78, ptr %57, align 8, !tbaa !5
  %79 = getelementptr inbounds nuw i8, ptr %10, i64 28
  %80 = load float, ptr %79, align 4, !tbaa !5
  %81 = fmul float %80, 0x400921FB60000000
  store float %81, ptr %58, align 4, !tbaa !5
  %82 = fpext float %81 to double
  store double %82, ptr %22, align 8, !tbaa !2
  %83 = fpext float %46 to double
  %84 = getelementptr inbounds nuw i8, ptr %22, i64 8
  store double %83, ptr %84, align 8, !tbaa !2
  %85 = fpext float %44 to double
  %86 = getelementptr inbounds nuw i8, ptr %22, i64 16
  store double %85, ptr %86, align 8, !tbaa !2
  %87 = fpext float %42 to double
  %88 = getelementptr inbounds nuw i8, ptr %22, i64 24
  store double %87, ptr %88, align 8, !tbaa !2
  %89 = getelementptr inbounds nuw i8, ptr %1, i64 80
  %90 = load float, ptr %89, align 4, !tbaa !5
  %91 = getelementptr inbounds nuw i8, ptr %1, i64 76
  %92 = load float, ptr %91, align 4, !tbaa !5
  %93 = getelementptr inbounds nuw i8, ptr %1, i64 72
  %94 = load float, ptr %93, align 4, !tbaa !5
  %95 = fpext float %78 to double
  %96 = getelementptr inbounds nuw i8, ptr %22, i64 32
  store double %95, ptr %96, align 8, !tbaa !2
  %97 = fpext float %94 to double
  %98 = getelementptr inbounds nuw i8, ptr %22, i64 40
  store double %97, ptr %98, align 8, !tbaa !2
  %99 = fpext float %92 to double
  %100 = getelementptr inbounds nuw i8, ptr %22, i64 48
  store double %99, ptr %100, align 8, !tbaa !2
  %101 = fpext float %90 to double
  %102 = getelementptr inbounds nuw i8, ptr %22, i64 56
  store double %101, ptr %102, align 8, !tbaa !2
  %103 = getelementptr inbounds nuw i8, ptr %1, i64 68
  %104 = load float, ptr %103, align 4, !tbaa !5
  %105 = getelementptr inbounds nuw i8, ptr %1, i64 64
  %106 = load float, ptr %105, align 4, !tbaa !5
  %107 = getelementptr inbounds nuw i8, ptr %1, i64 60
  %108 = load float, ptr %107, align 4, !tbaa !5
  %109 = fpext float %75 to double
  %110 = getelementptr inbounds nuw i8, ptr %22, i64 64
  store double %109, ptr %110, align 8, !tbaa !2
  %111 = fpext float %108 to double
  %112 = getelementptr inbounds nuw i8, ptr %22, i64 72
  store double %111, ptr %112, align 8, !tbaa !2
  %113 = fpext float %106 to double
  %114 = getelementptr inbounds nuw i8, ptr %22, i64 80
  store double %113, ptr %114, align 8, !tbaa !2
  %115 = fpext float %104 to double
  %116 = getelementptr inbounds nuw i8, ptr %22, i64 88
  store double %115, ptr %116, align 8, !tbaa !2
  %117 = getelementptr inbounds nuw i8, ptr %1, i64 56
  %118 = load float, ptr %117, align 4, !tbaa !5
  %119 = getelementptr inbounds nuw i8, ptr %1, i64 52
  %120 = load float, ptr %119, align 4, !tbaa !5
  %121 = getelementptr inbounds nuw i8, ptr %1, i64 48
  %122 = load float, ptr %121, align 4, !tbaa !5
  %123 = fpext float %72 to double
  %124 = getelementptr inbounds nuw i8, ptr %22, i64 96
  store double %123, ptr %124, align 8, !tbaa !2
  %125 = fpext float %122 to double
  %126 = getelementptr inbounds nuw i8, ptr %22, i64 104
  store double %125, ptr %126, align 8, !tbaa !2
  %127 = fpext float %120 to double
  %128 = getelementptr inbounds nuw i8, ptr %22, i64 112
  store double %127, ptr %128, align 8, !tbaa !2
  %129 = fpext float %118 to double
  %130 = getelementptr inbounds nuw i8, ptr %22, i64 120
  store double %129, ptr %130, align 8, !tbaa !2
  %131 = getelementptr inbounds nuw i8, ptr %1, i64 44
  %132 = load float, ptr %131, align 4, !tbaa !5
  %133 = getelementptr inbounds nuw i8, ptr %1, i64 40
  %134 = load float, ptr %133, align 4, !tbaa !5
  %135 = getelementptr inbounds nuw i8, ptr %1, i64 36
  %136 = load float, ptr %135, align 4, !tbaa !5
  %137 = fpext float %69 to double
  %138 = getelementptr inbounds nuw i8, ptr %22, i64 128
  store double %137, ptr %138, align 8, !tbaa !2
  %139 = fpext float %136 to double
  %140 = getelementptr inbounds nuw i8, ptr %22, i64 136
  store double %139, ptr %140, align 8, !tbaa !2
  %141 = fpext float %134 to double
  %142 = getelementptr inbounds nuw i8, ptr %22, i64 144
  store double %141, ptr %142, align 8, !tbaa !2
  %143 = fpext float %132 to double
  %144 = getelementptr inbounds nuw i8, ptr %22, i64 152
  store double %143, ptr %144, align 8, !tbaa !2
  %145 = getelementptr inbounds nuw i8, ptr %1, i64 32
  %146 = load float, ptr %145, align 4, !tbaa !5
  %147 = getelementptr inbounds nuw i8, ptr %1, i64 28
  %148 = load float, ptr %147, align 4, !tbaa !5
  %149 = getelementptr inbounds nuw i8, ptr %1, i64 24
  %150 = load float, ptr %149, align 4, !tbaa !5
  %151 = fpext float %66 to double
  %152 = getelementptr inbounds nuw i8, ptr %22, i64 160
  store double %151, ptr %152, align 8, !tbaa !2
  %153 = fpext float %150 to double
  %154 = getelementptr inbounds nuw i8, ptr %22, i64 168
  store double %153, ptr %154, align 8, !tbaa !2
  %155 = fpext float %148 to double
  %156 = getelementptr inbounds nuw i8, ptr %22, i64 176
  store double %155, ptr %156, align 8, !tbaa !2
  %157 = fpext float %146 to double
  %158 = getelementptr inbounds nuw i8, ptr %22, i64 184
  store double %157, ptr %158, align 8, !tbaa !2
  %159 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %160 = load float, ptr %159, align 4, !tbaa !5
  %161 = getelementptr inbounds nuw i8, ptr %1, i64 4
  %162 = load float, ptr %161, align 4, !tbaa !5
  %163 = load float, ptr %1, align 4, !tbaa !5
  %164 = fpext float %60 to double
  %165 = getelementptr inbounds nuw i8, ptr %22, i64 192
  store double %164, ptr %165, align 8, !tbaa !2
  %166 = fpext float %163 to double
  %167 = getelementptr inbounds nuw i8, ptr %22, i64 200
  store double %166, ptr %167, align 8, !tbaa !2
  %168 = fpext float %162 to double
  %169 = getelementptr inbounds nuw i8, ptr %22, i64 208
  store double %168, ptr %169, align 8, !tbaa !2
  %170 = fpext float %160 to double
  %171 = getelementptr inbounds nuw i8, ptr %22, i64 216
  store double %170, ptr %171, align 8, !tbaa !2
  %172 = getelementptr inbounds nuw i8, ptr %1, i64 20
  %173 = load float, ptr %172, align 4, !tbaa !5
  %174 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %175 = load float, ptr %174, align 4, !tbaa !5
  %176 = getelementptr inbounds nuw i8, ptr %1, i64 12
  %177 = load float, ptr %176, align 4, !tbaa !5
  tail call void @_mlir_memref_to_llvm_free(ptr %47)
  %178 = fpext float %63 to double
  %179 = getelementptr inbounds nuw i8, ptr %22, i64 224
  store double %178, ptr %179, align 8, !tbaa !2
  %180 = fpext float %177 to double
  %181 = getelementptr inbounds nuw i8, ptr %22, i64 232
  store double %180, ptr %181, align 8, !tbaa !2
  %182 = fpext float %175 to double
  %183 = getelementptr inbounds nuw i8, ptr %22, i64 240
  store double %182, ptr %183, align 8, !tbaa !2
  %184 = fpext float %173 to double
  %185 = getelementptr inbounds nuw i8, ptr %22, i64 248
  store double %184, ptr %185, align 8, !tbaa !2
  %186 = fpext float %40 to double
  %187 = getelementptr inbounds nuw i8, ptr %22, i64 256
  store double %186, ptr %187, align 8, !tbaa !2
  %188 = fpext float %38 to double
  %189 = getelementptr inbounds nuw i8, ptr %22, i64 264
  store double %188, ptr %189, align 8, !tbaa !2
  %190 = fpext float %36 to double
  %191 = getelementptr inbounds nuw i8, ptr %22, i64 272
  store double %190, ptr %191, align 8, !tbaa !2
  %192 = getelementptr inbounds nuw i8, ptr %1, i64 152
  %193 = load float, ptr %192, align 4, !tbaa !5
  %194 = getelementptr inbounds nuw i8, ptr %1, i64 148
  %195 = load float, ptr %194, align 4, !tbaa !5
  %196 = getelementptr inbounds nuw i8, ptr %1, i64 144
  %197 = load float, ptr %196, align 4, !tbaa !5
  %198 = fpext float %197 to double
  %199 = getelementptr inbounds nuw i8, ptr %22, i64 280
  store double %198, ptr %199, align 8, !tbaa !2
  %200 = fpext float %195 to double
  %201 = getelementptr inbounds nuw i8, ptr %22, i64 288
  store double %200, ptr %201, align 8, !tbaa !2
  %202 = fpext float %193 to double
  %203 = getelementptr inbounds nuw i8, ptr %22, i64 296
  store double %202, ptr %203, align 8, !tbaa !2
  %204 = getelementptr inbounds nuw i8, ptr %1, i64 104
  %205 = load float, ptr %204, align 4, !tbaa !5
  %206 = getelementptr inbounds nuw i8, ptr %1, i64 100
  %207 = load float, ptr %206, align 4, !tbaa !5
  %208 = getelementptr inbounds nuw i8, ptr %1, i64 96
  %209 = load float, ptr %208, align 4, !tbaa !5
  %210 = fpext float %209 to double
  %211 = getelementptr inbounds nuw i8, ptr %22, i64 304
  store double %210, ptr %211, align 8, !tbaa !2
  %212 = fpext float %207 to double
  %213 = getelementptr inbounds nuw i8, ptr %22, i64 312
  store double %212, ptr %213, align 8, !tbaa !2
  %214 = fpext float %205 to double
  %215 = getelementptr inbounds nuw i8, ptr %22, i64 320
  store double %214, ptr %215, align 8, !tbaa !2
  %216 = getelementptr inbounds nuw i8, ptr %1, i64 128
  %217 = load float, ptr %216, align 4, !tbaa !5
  %218 = getelementptr inbounds nuw i8, ptr %1, i64 124
  %219 = load float, ptr %218, align 4, !tbaa !5
  %220 = getelementptr inbounds nuw i8, ptr %1, i64 120
  %221 = load float, ptr %220, align 4, !tbaa !5
  %222 = fpext float %221 to double
  %223 = getelementptr inbounds nuw i8, ptr %22, i64 328
  store double %222, ptr %223, align 8, !tbaa !2
  %224 = fpext float %219 to double
  %225 = getelementptr inbounds nuw i8, ptr %22, i64 336
  store double %224, ptr %225, align 8, !tbaa !2
  %226 = fpext float %217 to double
  %227 = getelementptr inbounds nuw i8, ptr %22, i64 344
  store double %226, ptr %227, align 8, !tbaa !2
  %228 = fpext float %34 to double
  %229 = getelementptr inbounds nuw i8, ptr %22, i64 352
  store double %228, ptr %229, align 8, !tbaa !2
  %230 = fpext float %32 to double
  %231 = getelementptr inbounds nuw i8, ptr %22, i64 360
  store double %230, ptr %231, align 8, !tbaa !2
  %232 = fpext float %30 to double
  %233 = getelementptr inbounds nuw i8, ptr %22, i64 368
  store double %232, ptr %233, align 8, !tbaa !2
  %234 = getelementptr inbounds nuw i8, ptr %1, i64 236
  %235 = load float, ptr %234, align 4, !tbaa !5
  %236 = getelementptr inbounds nuw i8, ptr %1, i64 232
  %237 = load float, ptr %236, align 4, !tbaa !5
  %238 = getelementptr inbounds nuw i8, ptr %1, i64 228
  %239 = load float, ptr %238, align 4, !tbaa !5
  %240 = getelementptr inbounds nuw i8, ptr %1, i64 164
  %241 = load float, ptr %240, align 4, !tbaa !5
  %242 = getelementptr inbounds nuw i8, ptr %1, i64 160
  %243 = load float, ptr %242, align 4, !tbaa !5
  %244 = getelementptr inbounds nuw i8, ptr %1, i64 156
  %245 = load float, ptr %244, align 4, !tbaa !5
  %246 = fpext float %245 to double
  %247 = getelementptr inbounds nuw i8, ptr %22, i64 376
  store double %246, ptr %247, align 8, !tbaa !2
  %248 = fpext float %243 to double
  %249 = getelementptr inbounds nuw i8, ptr %22, i64 384
  store double %248, ptr %249, align 8, !tbaa !2
  %250 = fpext float %241 to double
  %251 = getelementptr inbounds nuw i8, ptr %22, i64 392
  store double %250, ptr %251, align 8, !tbaa !2
  %252 = getelementptr inbounds nuw i8, ptr %1, i64 116
  %253 = load float, ptr %252, align 4, !tbaa !5
  %254 = getelementptr inbounds nuw i8, ptr %1, i64 112
  %255 = load float, ptr %254, align 4, !tbaa !5
  %256 = getelementptr inbounds nuw i8, ptr %1, i64 108
  %257 = load float, ptr %256, align 4, !tbaa !5
  %258 = fpext float %257 to double
  %259 = getelementptr inbounds nuw i8, ptr %22, i64 400
  store double %258, ptr %259, align 8, !tbaa !2
  %260 = fpext float %255 to double
  %261 = getelementptr inbounds nuw i8, ptr %22, i64 408
  store double %260, ptr %261, align 8, !tbaa !2
  %262 = fpext float %253 to double
  %263 = getelementptr inbounds nuw i8, ptr %22, i64 416
  store double %262, ptr %263, align 8, !tbaa !2
  %264 = getelementptr inbounds nuw i8, ptr %1, i64 140
  %265 = load float, ptr %264, align 4, !tbaa !5
  %266 = getelementptr inbounds nuw i8, ptr %1, i64 136
  %267 = load float, ptr %266, align 4, !tbaa !5
  %268 = getelementptr inbounds nuw i8, ptr %1, i64 132
  %269 = load float, ptr %268, align 4, !tbaa !5
  %270 = fpext float %269 to double
  %271 = getelementptr inbounds nuw i8, ptr %22, i64 424
  store double %270, ptr %271, align 8, !tbaa !2
  %272 = fpext float %267 to double
  %273 = getelementptr inbounds nuw i8, ptr %22, i64 432
  store double %272, ptr %273, align 8, !tbaa !2
  %274 = fpext float %265 to double
  %275 = getelementptr inbounds nuw i8, ptr %22, i64 440
  store double %274, ptr %275, align 8, !tbaa !2
  %276 = fpext float %239 to double
  %277 = getelementptr inbounds nuw i8, ptr %22, i64 448
  store double %276, ptr %277, align 8, !tbaa !2
  %278 = fpext float %237 to double
  %279 = getelementptr inbounds nuw i8, ptr %22, i64 456
  store double %278, ptr %279, align 8, !tbaa !2
  %280 = fpext float %235 to double
  %281 = getelementptr inbounds nuw i8, ptr %22, i64 464
  store double %280, ptr %281, align 8, !tbaa !2
  %282 = getelementptr inbounds nuw i8, ptr %1, i64 224
  %283 = load float, ptr %282, align 4, !tbaa !5
  %284 = getelementptr inbounds nuw i8, ptr %1, i64 220
  %285 = load float, ptr %284, align 4, !tbaa !5
  %286 = getelementptr inbounds nuw i8, ptr %1, i64 216
  %287 = load float, ptr %286, align 4, !tbaa !5
  %288 = fpext float %287 to double
  %289 = getelementptr inbounds nuw i8, ptr %22, i64 472
  store double %288, ptr %289, align 8, !tbaa !2
  %290 = fpext float %285 to double
  %291 = getelementptr inbounds nuw i8, ptr %22, i64 480
  store double %290, ptr %291, align 8, !tbaa !2
  %292 = fpext float %283 to double
  %293 = getelementptr inbounds nuw i8, ptr %22, i64 488
  store double %292, ptr %293, align 8, !tbaa !2
  %294 = getelementptr inbounds nuw i8, ptr %1, i64 260
  %295 = load float, ptr %294, align 4, !tbaa !5
  %296 = getelementptr inbounds nuw i8, ptr %1, i64 256
  %297 = load float, ptr %296, align 4, !tbaa !5
  %298 = getelementptr inbounds nuw i8, ptr %1, i64 252
  %299 = load float, ptr %298, align 4, !tbaa !5
  %300 = getelementptr inbounds nuw i8, ptr %1, i64 188
  %301 = load float, ptr %300, align 4, !tbaa !5
  %302 = getelementptr inbounds nuw i8, ptr %1, i64 184
  %303 = load float, ptr %302, align 4, !tbaa !5
  %304 = getelementptr inbounds nuw i8, ptr %1, i64 180
  %305 = load float, ptr %304, align 4, !tbaa !5
  %306 = fpext float %305 to double
  %307 = getelementptr inbounds nuw i8, ptr %22, i64 496
  store double %306, ptr %307, align 8, !tbaa !2
  %308 = fpext float %303 to double
  %309 = getelementptr inbounds nuw i8, ptr %22, i64 504
  store double %308, ptr %309, align 8, !tbaa !2
  %310 = fpext float %301 to double
  %311 = getelementptr inbounds nuw i8, ptr %22, i64 512
  store double %310, ptr %311, align 8, !tbaa !2
  %312 = fpext float %299 to double
  %313 = getelementptr inbounds nuw i8, ptr %22, i64 520
  store double %312, ptr %313, align 8, !tbaa !2
  %314 = fpext float %297 to double
  %315 = getelementptr inbounds nuw i8, ptr %22, i64 528
  store double %314, ptr %315, align 8, !tbaa !2
  %316 = fpext float %295 to double
  %317 = getelementptr inbounds nuw i8, ptr %22, i64 536
  store double %316, ptr %317, align 8, !tbaa !2
  %318 = fpext float %28 to double
  %319 = getelementptr inbounds nuw i8, ptr %22, i64 544
  store double %318, ptr %319, align 8, !tbaa !2
  %320 = fpext float %26 to double
  %321 = getelementptr inbounds nuw i8, ptr %22, i64 552
  store double %320, ptr %321, align 8, !tbaa !2
  %322 = fpext float %24 to double
  %323 = getelementptr inbounds nuw i8, ptr %22, i64 560
  store double %322, ptr %323, align 8, !tbaa !2
  %324 = getelementptr inbounds nuw i8, ptr %1, i64 344
  %325 = load float, ptr %324, align 4, !tbaa !5
  %326 = getelementptr inbounds nuw i8, ptr %1, i64 340
  %327 = load float, ptr %326, align 4, !tbaa !5
  %328 = getelementptr inbounds nuw i8, ptr %1, i64 336
  %329 = load float, ptr %328, align 4, !tbaa !5
  %330 = getelementptr inbounds nuw i8, ptr %1, i64 284
  %331 = load float, ptr %330, align 4, !tbaa !5
  %332 = getelementptr inbounds nuw i8, ptr %1, i64 280
  %333 = load float, ptr %332, align 4, !tbaa !5
  %334 = getelementptr inbounds nuw i8, ptr %1, i64 276
  %335 = load float, ptr %334, align 4, !tbaa !5
  %336 = fpext float %335 to double
  %337 = getelementptr inbounds nuw i8, ptr %22, i64 568
  store double %336, ptr %337, align 8, !tbaa !2
  %338 = fpext float %333 to double
  %339 = getelementptr inbounds nuw i8, ptr %22, i64 576
  store double %338, ptr %339, align 8, !tbaa !2
  %340 = fpext float %331 to double
  %341 = getelementptr inbounds nuw i8, ptr %22, i64 584
  store double %340, ptr %341, align 8, !tbaa !2
  %342 = getelementptr inbounds nuw i8, ptr %1, i64 212
  %343 = load float, ptr %342, align 4, !tbaa !5
  %344 = getelementptr inbounds nuw i8, ptr %1, i64 208
  %345 = load float, ptr %344, align 4, !tbaa !5
  %346 = getelementptr inbounds nuw i8, ptr %1, i64 204
  %347 = load float, ptr %346, align 4, !tbaa !5
  %348 = fpext float %347 to double
  %349 = getelementptr inbounds nuw i8, ptr %22, i64 592
  store double %348, ptr %349, align 8, !tbaa !2
  %350 = fpext float %345 to double
  %351 = getelementptr inbounds nuw i8, ptr %22, i64 600
  store double %350, ptr %351, align 8, !tbaa !2
  %352 = fpext float %343 to double
  %353 = getelementptr inbounds nuw i8, ptr %22, i64 608
  store double %352, ptr %353, align 8, !tbaa !2
  %354 = getelementptr inbounds nuw i8, ptr %1, i64 248
  %355 = load float, ptr %354, align 4, !tbaa !5
  %356 = getelementptr inbounds nuw i8, ptr %1, i64 244
  %357 = load float, ptr %356, align 4, !tbaa !5
  %358 = getelementptr inbounds nuw i8, ptr %1, i64 240
  %359 = load float, ptr %358, align 4, !tbaa !5
  %360 = fpext float %359 to double
  %361 = getelementptr inbounds nuw i8, ptr %22, i64 616
  store double %360, ptr %361, align 8, !tbaa !2
  %362 = fpext float %357 to double
  %363 = getelementptr inbounds nuw i8, ptr %22, i64 624
  store double %362, ptr %363, align 8, !tbaa !2
  %364 = fpext float %355 to double
  %365 = getelementptr inbounds nuw i8, ptr %22, i64 632
  store double %364, ptr %365, align 8, !tbaa !2
  %366 = fpext float %329 to double
  %367 = getelementptr inbounds nuw i8, ptr %22, i64 640
  store double %366, ptr %367, align 8, !tbaa !2
  %368 = fpext float %327 to double
  %369 = getelementptr inbounds nuw i8, ptr %22, i64 648
  store double %368, ptr %369, align 8, !tbaa !2
  %370 = fpext float %325 to double
  %371 = getelementptr inbounds nuw i8, ptr %22, i64 656
  store double %370, ptr %371, align 8, !tbaa !2
  %372 = getelementptr inbounds nuw i8, ptr %1, i64 308
  %373 = load float, ptr %372, align 4, !tbaa !5
  %374 = getelementptr inbounds nuw i8, ptr %1, i64 304
  %375 = load float, ptr %374, align 4, !tbaa !5
  %376 = getelementptr inbounds nuw i8, ptr %1, i64 300
  %377 = load float, ptr %376, align 4, !tbaa !5
  %378 = getelementptr inbounds nuw i8, ptr %1, i64 272
  %379 = load float, ptr %378, align 4, !tbaa !5
  %380 = getelementptr inbounds nuw i8, ptr %1, i64 268
  %381 = load float, ptr %380, align 4, !tbaa !5
  %382 = getelementptr inbounds nuw i8, ptr %1, i64 264
  %383 = load float, ptr %382, align 4, !tbaa !5
  %384 = fpext float %383 to double
  %385 = getelementptr inbounds nuw i8, ptr %22, i64 664
  store double %384, ptr %385, align 8, !tbaa !2
  %386 = fpext float %381 to double
  %387 = getelementptr inbounds nuw i8, ptr %22, i64 672
  store double %386, ptr %387, align 8, !tbaa !2
  %388 = fpext float %379 to double
  %389 = getelementptr inbounds nuw i8, ptr %22, i64 680
  store double %388, ptr %389, align 8, !tbaa !2
  %390 = fpext float %377 to double
  %391 = getelementptr inbounds nuw i8, ptr %22, i64 688
  store double %390, ptr %391, align 8, !tbaa !2
  %392 = fpext float %375 to double
  %393 = getelementptr inbounds nuw i8, ptr %22, i64 696
  store double %392, ptr %393, align 8, !tbaa !2
  %394 = fpext float %373 to double
  %395 = getelementptr inbounds nuw i8, ptr %22, i64 704
  store double %394, ptr %395, align 8, !tbaa !2
  %396 = getelementptr inbounds nuw i8, ptr %1, i64 356
  %397 = load float, ptr %396, align 4, !tbaa !5
  %398 = getelementptr inbounds nuw i8, ptr %1, i64 352
  %399 = load float, ptr %398, align 4, !tbaa !5
  %400 = getelementptr inbounds nuw i8, ptr %1, i64 348
  %401 = load float, ptr %400, align 4, !tbaa !5
  %402 = fpext float %401 to double
  %403 = getelementptr inbounds nuw i8, ptr %22, i64 712
  store double %402, ptr %403, align 8, !tbaa !2
  %404 = fpext float %399 to double
  %405 = getelementptr inbounds nuw i8, ptr %22, i64 720
  store double %404, ptr %405, align 8, !tbaa !2
  %406 = fpext float %397 to double
  %407 = getelementptr inbounds nuw i8, ptr %22, i64 728
  store double %406, ptr %407, align 8, !tbaa !2
  %408 = getelementptr inbounds nuw i8, ptr %1, i64 320
  %409 = load float, ptr %408, align 4, !tbaa !5
  %410 = getelementptr inbounds nuw i8, ptr %1, i64 316
  %411 = load float, ptr %410, align 4, !tbaa !5
  %412 = getelementptr inbounds nuw i8, ptr %1, i64 312
  %413 = load float, ptr %412, align 4, !tbaa !5
  %414 = fpext float %413 to double
  %415 = getelementptr inbounds nuw i8, ptr %22, i64 736
  store double %414, ptr %415, align 8, !tbaa !2
  %416 = fpext float %411 to double
  %417 = getelementptr inbounds nuw i8, ptr %22, i64 744
  store double %416, ptr %417, align 8, !tbaa !2
  %418 = fpext float %409 to double
  %419 = getelementptr inbounds nuw i8, ptr %22, i64 752
  store double %418, ptr %419, align 8, !tbaa !2
  %420 = getelementptr inbounds nuw i8, ptr %1, i64 368
  %421 = load float, ptr %420, align 4, !tbaa !5
  %422 = getelementptr inbounds nuw i8, ptr %1, i64 364
  %423 = load float, ptr %422, align 4, !tbaa !5
  %424 = getelementptr inbounds nuw i8, ptr %1, i64 360
  %425 = load float, ptr %424, align 4, !tbaa !5
  %426 = fpext float %425 to double
  %427 = getelementptr inbounds nuw i8, ptr %22, i64 760
  store double %426, ptr %427, align 8, !tbaa !2
  %428 = fpext float %423 to double
  %429 = getelementptr inbounds nuw i8, ptr %22, i64 768
  store double %428, ptr %429, align 8, !tbaa !2
  %430 = fpext float %421 to double
  %431 = getelementptr inbounds nuw i8, ptr %22, i64 776
  store double %430, ptr %431, align 8, !tbaa !2
  %432 = getelementptr inbounds nuw i8, ptr %1, i64 332
  %433 = load float, ptr %432, align 4, !tbaa !5
  %434 = getelementptr inbounds nuw i8, ptr %1, i64 328
  %435 = load float, ptr %434, align 4, !tbaa !5
  %436 = getelementptr inbounds nuw i8, ptr %1, i64 324
  %437 = load float, ptr %436, align 4, !tbaa !5
  %438 = fpext float %437 to double
  %439 = getelementptr inbounds nuw i8, ptr %22, i64 784
  store double %438, ptr %439, align 8, !tbaa !2
  %440 = fpext float %435 to double
  %441 = getelementptr inbounds nuw i8, ptr %22, i64 792
  store double %440, ptr %441, align 8, !tbaa !2
  %442 = fpext float %433 to double
  %443 = getelementptr inbounds nuw i8, ptr %22, i64 800
  store double %442, ptr %443, align 8, !tbaa !2
  %444 = getelementptr inbounds nuw i8, ptr %1, i64 380
  %445 = load float, ptr %444, align 4, !tbaa !5
  %446 = getelementptr inbounds nuw i8, ptr %1, i64 376
  %447 = load float, ptr %446, align 4, !tbaa !5
  %448 = getelementptr inbounds nuw i8, ptr %1, i64 372
  %449 = load float, ptr %448, align 4, !tbaa !5
  %450 = fpext float %449 to double
  %451 = getelementptr inbounds nuw i8, ptr %22, i64 808
  store double %450, ptr %451, align 8, !tbaa !2
  %452 = fpext float %447 to double
  %453 = getelementptr inbounds nuw i8, ptr %22, i64 816
  store double %452, ptr %453, align 8, !tbaa !2
  %454 = fpext float %445 to double
  %455 = getelementptr inbounds nuw i8, ptr %22, i64 824
  store double %454, ptr %455, align 8, !tbaa !2
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
  %456 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 8)
  store ptr %456, ptr %18, align 8
  %.fca.1.gep127 = getelementptr inbounds nuw i8, ptr %18, i64 8
  store ptr %456, ptr %.fca.1.gep127, align 8
  %.fca.2.gep129 = getelementptr inbounds nuw i8, ptr %18, i64 16
  store i64 0, ptr %.fca.2.gep129, align 8
  call void @qnode_forward_0.quantum(ptr nonnull %21, ptr nonnull %20, ptr nonnull %19, ptr nonnull %18)
  %457 = load double, ptr %456, align 8, !tbaa !2
  store double %457, ptr %16, align 8, !tbaa !2
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
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #5

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #6

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #7

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #7

attributes #0 = { "enzyme_math"="free" "prev_linkage"="0" }
attributes #1 = { "enzyme_allocator"="0" "enzyme_deallocator"="-1" "prev_linkage"="0" }
attributes #2 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) }
attributes #3 = { "prev_linkage"="0" }
attributes #4 = { noinline "prev_linkage"="0" }
attributes #5 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #6 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #7 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #8 = { mustprogress willreturn }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{ptr @_mlir_memref_to_llvm_free}
!2 = !{!3, !3, i64 0}
!3 = !{!"double", !4, i64 0}
!4 = !{!"Catalyst TBAA"}
!5 = !{!6, !6, i64 0}
!6 = !{!"float", !4, i64 0}
!7 = !{!8}
!8 = distinct !{!8, !9, !"primal"}
!9 = distinct !{!9, !" diff: %"}
!10 = !{!11}
!11 = distinct !{!11, !9, !"shadow_0"}
!12 = !{!13}
!13 = distinct !{!13, !14, !"primal"}
!14 = distinct !{!14, !" diff: %"}
!15 = !{!16}
!16 = distinct !{!16, !14, !"shadow_0"}
!17 = !{!18}
!18 = distinct !{!18, !19, !"primal"}
!19 = distinct !{!19, !" diff: %"}
!20 = !{!21}
!21 = distinct !{!21, !19, !"shadow_0"}
!22 = !{!23}
!23 = distinct !{!23, !24, !"primal"}
!24 = distinct !{!24, !" diff: %"}
!25 = !{!26}
!26 = distinct !{!26, !24, !"shadow_0"}
!27 = !{!28}
!28 = distinct !{!28, !29, !"primal"}
!29 = distinct !{!29, !" diff: %"}
!30 = !{!31}
!31 = distinct !{!31, !29, !"shadow_0"}
!32 = !{!33}
!33 = distinct !{!33, !34, !"shadow_0"}
!34 = distinct !{!34, !" diff: %"}
!35 = !{!36}
!36 = distinct !{!36, !34, !"primal"}
!37 = !{!38}
!38 = distinct !{!38, !39, !"shadow_0"}
!39 = distinct !{!39, !" diff: %"}
!40 = !{!41}
!41 = distinct !{!41, !39, !"primal"}
!42 = !{!43}
!43 = distinct !{!43, !44, !"shadow_0"}
!44 = distinct !{!44, !" diff: %"}
!45 = !{!46}
!46 = distinct !{!46, !44, !"primal"}
!47 = !{!48}
!48 = distinct !{!48, !49, !"shadow_0"}
!49 = distinct !{!49, !" diff: %"}
!50 = !{!51}
!51 = distinct !{!51, !49, !"primal"}
!52 = !{ptr @qnode_forward_0.quantum.augfwd}
!53 = !{ptr @qnode_forward_0.quantum.customqgrad}
