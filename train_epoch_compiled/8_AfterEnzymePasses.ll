; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@"{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}" = internal constant [54 x i8] c"{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}\00"
@LightningSimulator = internal constant [19 x i8] c"LightningSimulator\00"
@"/home/pichau/QAgents/.venv/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.so" = internal constant [107 x i8] c"/home/pichau/QAgents/.venv/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.so\00"
@__constant_4xi32_3 = private unnamed_addr constant [4 x i32] [i32 13, i32 15, i32 26, i32 6], align 64
@__constant_4xi32 = private unnamed_addr constant [4 x i32] [i32 17, i32 29, i32 16, i32 24], align 64

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

define { { ptr, ptr, i64, [3 x i64], [3 x i64] }, { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64, [1 x i64], [1 x i64] }, { ptr, ptr, i64, [1 x i64], [1 x i64] } } @jit_train_epoch_compiled(ptr readnone captures(none) %0, ptr readonly captures(none) %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, ptr readnone captures(none) %9, ptr readonly captures(none) %10, i64 %11, ptr readnone captures(none) %12, ptr readonly captures(none) %13, i64 %14, ptr %15, ptr %16, i64 %17, ptr readnone captures(none) %18, ptr readonly captures(none) %19, i64 %20, i64 %21, i64 %22, ptr %23, ptr %24, i64 %25, i64 %26, i64 %27, i64 %28, i64 %29, ptr readnone captures(none) %30, ptr readonly captures(none) %31, i64 %32, i64 %33, i64 %34, ptr readnone captures(none) %35, ptr readonly captures(none) %36, i64 %37, i64 %38, i64 %39) local_unnamed_addr {
.preheader395.preheader:
  %40 = load i32, ptr %19, align 4, !tbaa !2
  %41 = getelementptr inbounds nuw i8, ptr %19, i64 4
  %42 = load i32, ptr %41, align 4, !tbaa !2
  %43 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 80)
  %44 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 80)
  %45 = ptrtoint ptr %44 to i64
  %46 = add i64 %45, 63
  %47 = and i64 %46, -64
  %48 = inttoptr i64 %47 to ptr
  store i64 0, ptr %48, align 64, !tbaa !2
  %49 = getelementptr inbounds nuw i8, ptr %48, i64 8
  store i64 1, ptr %49, align 8, !tbaa !2
  %50 = ptrtoint ptr %43 to i64
  %51 = add i64 %50, 63
  %52 = and i64 %51, -64
  %53 = inttoptr i64 %52 to ptr
  store i64 1, ptr %53, align 64, !tbaa !2
  %54 = getelementptr inbounds nuw i8, ptr %53, i64 8
  store i64 1, ptr %54, align 8, !tbaa !2
  %55 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 80)
  %56 = ptrtoint ptr %55 to i64
  %57 = add i64 %56, 63
  %58 = and i64 %57, -64
  %59 = inttoptr i64 %58 to ptr
  %60 = load i64, ptr %53, align 64, !tbaa !2
  %61 = load i64, ptr %48, align 64, !tbaa !2
  %62 = mul i64 %61, %60
  store i64 %62, ptr %59, align 64, !tbaa !2
  %63 = load i64, ptr %54, align 8, !tbaa !2
  %64 = load i64, ptr %49, align 8, !tbaa !2
  %65 = mul i64 %64, %63
  %66 = getelementptr inbounds nuw i8, ptr %59, i64 8
  store i64 %65, ptr %66, align 8, !tbaa !2
  tail call void @_mlir_memref_to_llvm_free(ptr %44)
  store i64 32, ptr %53, align 64, !tbaa !2
  store i64 32, ptr %54, align 8, !tbaa !2
  %67 = load i64, ptr %59, align 64, !tbaa !2
  %68 = lshr i64 %67, 32
  store i64 %68, ptr %53, align 64, !tbaa !2
  %69 = load i64, ptr %66, align 8, !tbaa !2
  %70 = lshr i64 %69, 32
  store i64 %70, ptr %54, align 8, !tbaa !2
  %71 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %72 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %73 = ptrtoint ptr %72 to i64
  %74 = add i64 %73, 63
  %75 = and i64 %74, -64
  %76 = inttoptr i64 %75 to ptr
  %77 = load i64, ptr %59, align 64, !tbaa !2
  %78 = trunc i64 %77 to i32
  store i32 %78, ptr %76, align 64, !tbaa !2
  %79 = load i64, ptr %66, align 8, !tbaa !2
  %80 = trunc i64 %79 to i32
  %81 = getelementptr inbounds nuw i8, ptr %76, i64 4
  store i32 %80, ptr %81, align 4, !tbaa !2
  tail call void @_mlir_memref_to_llvm_free(ptr %55)
  %82 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %83 = ptrtoint ptr %82 to i64
  %84 = add i64 %83, 63
  %85 = and i64 %84, -64
  %86 = inttoptr i64 %85 to ptr
  %87 = load i64, ptr %53, align 64, !tbaa !2
  %88 = trunc i64 %87 to i32
  store i32 %88, ptr %86, align 64, !tbaa !2
  %89 = load i64, ptr %54, align 8, !tbaa !2
  %90 = trunc i64 %89 to i32
  %91 = getelementptr inbounds nuw i8, ptr %86, i64 4
  store i32 %90, ptr %91, align 4, !tbaa !2
  %92 = ptrtoint ptr %71 to i64
  %93 = add i64 %92, 63
  %94 = and i64 %93, -64
  %95 = inttoptr i64 %94 to ptr
  tail call void @_mlir_memref_to_llvm_free(ptr %43)
  %96 = load i32, ptr %19, align 4, !tbaa !2
  store i32 %96, ptr %95, align 64, !tbaa !2
  %97 = getelementptr inbounds nuw i8, ptr %95, i64 4
  store i32 %96, ptr %97, align 4, !tbaa !2
  %98 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %99 = ptrtoint ptr %98 to i64
  %100 = add i64 %99, 63
  %101 = and i64 %100, -64
  %102 = inttoptr i64 %101 to ptr
  %103 = load i32, ptr %86, align 64, !tbaa !2
  %104 = load i32, ptr %95, align 64, !tbaa !2
  %105 = add i32 %104, %103
  store i32 %105, ptr %102, align 64, !tbaa !2
  %106 = load i32, ptr %91, align 4, !tbaa !2
  %107 = load i32, ptr %97, align 4, !tbaa !2
  %108 = add i32 %107, %106
  %109 = getelementptr inbounds nuw i8, ptr %102, i64 4
  store i32 %108, ptr %109, align 4, !tbaa !2
  tail call void @_mlir_memref_to_llvm_free(ptr %82)
  %110 = load i32, ptr %41, align 4, !tbaa !2
  store i32 %110, ptr %95, align 64, !tbaa !2
  store i32 %110, ptr %97, align 4, !tbaa !2
  %111 = load i32, ptr %76, align 64, !tbaa !2
  %112 = add i32 %111, %110
  store i32 %112, ptr %95, align 64, !tbaa !2
  %113 = load i32, ptr %81, align 4, !tbaa !2
  %114 = add i32 %113, %110
  store i32 %114, ptr %97, align 4, !tbaa !2
  %115 = xor i32 %40, %42
  %116 = xor i32 %115, 466688986
  tail call void @_mlir_memref_to_llvm_free(ptr %72)
  %117 = load i32, ptr %41, align 4, !tbaa !2
  %118 = load i32, ptr %19, align 4, !tbaa !2
  %119 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 80)
  %120 = ptrtoint ptr %119 to i64
  %121 = add i64 %120, 63
  %122 = and i64 %121, -64
  %123 = inttoptr i64 %122 to ptr
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 64 dereferenceable(16) %123, ptr noundef nonnull align 64 dereferenceable(16) @__constant_4xi32_3, i64 16, i1 false)
  %124 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 80)
  %125 = ptrtoint ptr %124 to i64
  %126 = add i64 %125, 63
  %127 = and i64 %126, -64
  %128 = inttoptr i64 %127 to ptr
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 64 dereferenceable(16) %128, ptr noundef nonnull align 64 dereferenceable(16) @__constant_4xi32, i64 16, i1 false)
  %129 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 68)
  %130 = ptrtoint ptr %129 to i64
  %131 = add i64 %130, 63
  %132 = and i64 %131, -64
  %133 = inttoptr i64 %132 to ptr
  %134 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 68)
  %135 = ptrtoint ptr %134 to i64
  %136 = add i64 %135, 63
  %137 = and i64 %136, -64
  %138 = inttoptr i64 %137 to ptr
  %139 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %140 = ptrtoint ptr %139 to i64
  %141 = add i64 %140, 63
  %142 = and i64 %141, -64
  %143 = inttoptr i64 %142 to ptr
  %144 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %145 = ptrtoint ptr %144 to i64
  %146 = add i64 %145, 63
  %147 = and i64 %146, -64
  %148 = inttoptr i64 %147 to ptr
  %149 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 68)
  %150 = ptrtoint ptr %149 to i64
  %151 = add i64 %150, 63
  %152 = and i64 %151, -64
  %153 = inttoptr i64 %152 to ptr
  %154 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %155 = ptrtoint ptr %154 to i64
  %156 = add i64 %155, 63
  %157 = and i64 %156, -64
  %158 = inttoptr i64 %157 to ptr
  %159 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %160 = ptrtoint ptr %159 to i64
  %161 = add i64 %160, 63
  %162 = and i64 %161, -64
  %163 = inttoptr i64 %162 to ptr
  %164 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %165 = ptrtoint ptr %164 to i64
  %166 = add i64 %165, 63
  %167 = and i64 %166, -64
  %168 = inttoptr i64 %167 to ptr
  %169 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 68)
  %170 = ptrtoint ptr %169 to i64
  %171 = add i64 %170, 63
  %172 = and i64 %171, -64
  %173 = inttoptr i64 %172 to ptr
  %174 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %175 = ptrtoint ptr %174 to i64
  %176 = add i64 %175, 63
  %177 = and i64 %176, -64
  %178 = inttoptr i64 %177 to ptr
  %179 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %180 = ptrtoint ptr %179 to i64
  %181 = add i64 %180, 63
  %182 = and i64 %181, -64
  %183 = inttoptr i64 %182 to ptr
  %184 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %185 = ptrtoint ptr %184 to i64
  %186 = add i64 %185, 63
  %187 = and i64 %186, -64
  %188 = inttoptr i64 %187 to ptr
  %189 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 68)
  %190 = ptrtoint ptr %189 to i64
  %191 = add i64 %190, 63
  %192 = and i64 %191, -64
  %193 = inttoptr i64 %192 to ptr
  %194 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %195 = ptrtoint ptr %194 to i64
  %196 = add i64 %195, 63
  %197 = and i64 %196, -64
  %198 = inttoptr i64 %197 to ptr
  %199 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %200 = ptrtoint ptr %199 to i64
  %201 = add i64 %200, 63
  %202 = and i64 %201, -64
  %203 = inttoptr i64 %202 to ptr
  %204 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %205 = ptrtoint ptr %204 to i64
  %206 = add i64 %205, 63
  %207 = and i64 %206, -64
  %208 = inttoptr i64 %207 to ptr
  %209 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 68)
  %210 = ptrtoint ptr %209 to i64
  %211 = add i64 %210, 63
  %212 = and i64 %211, -64
  %213 = inttoptr i64 %212 to ptr
  %214 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %215 = ptrtoint ptr %214 to i64
  %216 = add i64 %215, 63
  %217 = and i64 %216, -64
  %218 = inttoptr i64 %217 to ptr
  %219 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %220 = ptrtoint ptr %219 to i64
  %221 = add i64 %220, 63
  %222 = and i64 %221, -64
  %223 = inttoptr i64 %222 to ptr
  %224 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 68)
  %225 = ptrtoint ptr %224 to i64
  %226 = add i64 %225, 63
  %227 = and i64 %226, -64
  %228 = inttoptr i64 %227 to ptr
  %229 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 8)
  %230 = load i64, ptr %95, align 64
  store i64 %230, ptr %229, align 1
  tail call void @_mlir_memref_to_llvm_free(ptr %71)
  %231 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 8)
  %232 = load i64, ptr %102, align 64
  store i64 %232, ptr %231, align 1
  tail call void @_mlir_memref_to_llvm_free(ptr %98)
  %233 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 16)
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(16) %233, ptr noundef nonnull align 64 dereferenceable(16) %123, i64 16, i1 false)
  tail call void @_mlir_memref_to_llvm_free(ptr %119)
  %234 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 16)
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(16) %234, ptr noundef nonnull align 64 dereferenceable(16) %128, i64 16, i1 false)
  tail call void @_mlir_memref_to_llvm_free(ptr %124)
  %235 = getelementptr inbounds nuw i8, ptr %143, i64 4
  %236 = getelementptr inbounds nuw i8, ptr %148, i64 4
  %237 = getelementptr inbounds nuw i8, ptr %158, i64 4
  %238 = getelementptr inbounds nuw i8, ptr %163, i64 4
  %239 = getelementptr inbounds nuw i8, ptr %168, i64 4
  %240 = getelementptr inbounds nuw i8, ptr %178, i64 4
  %241 = getelementptr inbounds nuw i8, ptr %183, i64 4
  %242 = getelementptr inbounds nuw i8, ptr %188, i64 4
  %243 = getelementptr inbounds nuw i8, ptr %198, i64 4
  %244 = getelementptr inbounds nuw i8, ptr %203, i64 4
  %245 = getelementptr inbounds nuw i8, ptr %208, i64 4
  %246 = getelementptr inbounds nuw i8, ptr %218, i64 4
  %247 = getelementptr inbounds nuw i8, ptr %223, i64 4
  br label %.preheader392.preheader

.preheader392.preheader:                          ; preds = %.preheader395.preheader, %.preheader392.preheader
  %.pn143409 = phi ptr [ %234, %.preheader395.preheader ], [ %447, %.preheader392.preheader ]
  %.pn133408 = phi ptr [ %233, %.preheader395.preheader ], [ %446, %.preheader392.preheader ]
  %248 = phi i32 [ %118, %.preheader395.preheader ], [ %250, %.preheader392.preheader ]
  %249 = phi i32 [ %116, %.preheader395.preheader ], [ %248, %.preheader392.preheader ]
  %250 = phi i32 [ %117, %.preheader395.preheader ], [ %249, %.preheader392.preheader ]
  %.pn123407 = phi ptr [ %229, %.preheader395.preheader ], [ %442, %.preheader392.preheader ]
  %.pn113406 = phi ptr [ %231, %.preheader395.preheader ], [ %444, %.preheader392.preheader ]
  %251 = phi i32 [ 0, %.preheader395.preheader ], [ %426, %.preheader392.preheader ]
  %252 = phi i64 [ 0, %.preheader395.preheader ], [ %448, %.preheader392.preheader ]
  store i32 %250, ptr %133, align 64, !tbaa !2
  store i32 %249, ptr %138, align 64, !tbaa !2
  %253 = load i32, ptr %.pn133408, align 4, !tbaa !2
  %254 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %255 = load i32, ptr %.pn113406, align 4, !tbaa !2
  %256 = load i32, ptr %.pn123407, align 4, !tbaa !2
  %257 = add i32 %256, %255
  store i32 %257, ptr %143, align 64, !tbaa !2
  %258 = getelementptr inbounds nuw i8, ptr %.pn113406, i64 4
  %259 = load i32, ptr %258, align 4, !tbaa !2
  %260 = getelementptr inbounds nuw i8, ptr %.pn123407, i64 4
  %261 = load i32, ptr %260, align 4, !tbaa !2
  %262 = add i32 %261, %259
  store i32 %262, ptr %235, align 4, !tbaa !2
  %263 = ptrtoint ptr %254 to i64
  %264 = add i64 %263, 63
  %265 = and i64 %264, -64
  %266 = inttoptr i64 %265 to ptr
  tail call void @_mlir_memref_to_llvm_free(ptr nonnull %.pn113406)
  %267 = load i32, ptr %.pn133408, align 4, !tbaa !2
  store i32 %267, ptr %266, align 64, !tbaa !2
  %268 = getelementptr inbounds nuw i8, ptr %266, i64 4
  store i32 %267, ptr %268, align 4, !tbaa !2
  %269 = load i32, ptr %.pn123407, align 4, !tbaa !2
  %270 = shl i32 %269, %267
  %271 = icmp ult i32 %267, 32
  %272 = select i1 %271, i32 %270, i32 0
  store i32 %272, ptr %148, align 64, !tbaa !2
  %273 = load i32, ptr %260, align 4, !tbaa !2
  %274 = load i32, ptr %268, align 4, !tbaa !2
  %275 = shl i32 %273, %274
  %276 = icmp ult i32 %274, 32
  %277 = select i1 %276, i32 %275, i32 0
  store i32 %277, ptr %236, align 4, !tbaa !2
  %278 = sub i32 32, %253
  store i32 %278, ptr %153, align 64, !tbaa !2
  store i32 %278, ptr %266, align 64, !tbaa !2
  %279 = load i32, ptr %153, align 64, !tbaa !2
  store i32 %279, ptr %268, align 4, !tbaa !2
  %280 = load i32, ptr %.pn123407, align 4, !tbaa !2
  %281 = lshr i32 %280, %278
  %282 = icmp ult i32 %278, 32
  %283 = select i1 %282, i32 %281, i32 0
  store i32 %283, ptr %266, align 64, !tbaa !2
  %284 = load i32, ptr %260, align 4, !tbaa !2
  %285 = lshr i32 %284, %279
  %286 = icmp ult i32 %279, 32
  %287 = select i1 %286, i32 %285, i32 0
  store i32 %287, ptr %268, align 4, !tbaa !2
  tail call void @_mlir_memref_to_llvm_free(ptr nonnull %.pn123407)
  %288 = load i32, ptr %148, align 64, !tbaa !2
  %289 = load i32, ptr %266, align 64, !tbaa !2
  %290 = or i32 %289, %288
  store i32 %290, ptr %266, align 64, !tbaa !2
  %291 = load i32, ptr %236, align 4, !tbaa !2
  %292 = load i32, ptr %268, align 4, !tbaa !2
  %293 = or i32 %292, %291
  store i32 %293, ptr %268, align 4, !tbaa !2
  %294 = load i32, ptr %143, align 64, !tbaa !2
  %295 = xor i32 %294, %290
  store i32 %295, ptr %158, align 64, !tbaa !2
  %296 = load i32, ptr %235, align 4, !tbaa !2
  %297 = load i32, ptr %268, align 4, !tbaa !2
  %298 = xor i32 %297, %296
  store i32 %298, ptr %237, align 4, !tbaa !2
  %299 = getelementptr inbounds nuw i8, ptr %.pn133408, i64 4
  %300 = load i32, ptr %299, align 4, !tbaa !2
  %301 = load i32, ptr %143, align 64, !tbaa !2
  %302 = add i32 %301, %295
  store i32 %302, ptr %163, align 64, !tbaa !2
  %303 = load i32, ptr %235, align 4, !tbaa !2
  %304 = load i32, ptr %237, align 4, !tbaa !2
  %305 = add i32 %304, %303
  store i32 %305, ptr %238, align 4, !tbaa !2
  %306 = load i32, ptr %299, align 4, !tbaa !2
  store i32 %306, ptr %266, align 64, !tbaa !2
  store i32 %306, ptr %268, align 4, !tbaa !2
  %307 = load i32, ptr %158, align 64, !tbaa !2
  %308 = shl i32 %307, %306
  %309 = icmp ult i32 %306, 32
  %310 = select i1 %309, i32 %308, i32 0
  store i32 %310, ptr %168, align 64, !tbaa !2
  %311 = load i32, ptr %237, align 4, !tbaa !2
  %312 = load i32, ptr %268, align 4, !tbaa !2
  %313 = shl i32 %311, %312
  %314 = icmp ult i32 %312, 32
  %315 = select i1 %314, i32 %313, i32 0
  store i32 %315, ptr %239, align 4, !tbaa !2
  %316 = sub i32 32, %300
  store i32 %316, ptr %173, align 64, !tbaa !2
  store i32 %316, ptr %266, align 64, !tbaa !2
  %317 = load i32, ptr %173, align 64, !tbaa !2
  store i32 %317, ptr %268, align 4, !tbaa !2
  %318 = load i32, ptr %158, align 64, !tbaa !2
  %319 = lshr i32 %318, %316
  %320 = icmp ult i32 %316, 32
  %321 = select i1 %320, i32 %319, i32 0
  store i32 %321, ptr %266, align 64, !tbaa !2
  %322 = load i32, ptr %237, align 4, !tbaa !2
  %323 = lshr i32 %322, %317
  %324 = icmp ult i32 %317, 32
  %325 = select i1 %324, i32 %323, i32 0
  store i32 %325, ptr %268, align 4, !tbaa !2
  %326 = load i32, ptr %168, align 64, !tbaa !2
  %327 = or i32 %326, %321
  store i32 %327, ptr %266, align 64, !tbaa !2
  %328 = load i32, ptr %239, align 4, !tbaa !2
  %329 = or i32 %328, %325
  store i32 %329, ptr %268, align 4, !tbaa !2
  %330 = load i32, ptr %163, align 64, !tbaa !2
  %331 = xor i32 %330, %327
  store i32 %331, ptr %178, align 64, !tbaa !2
  %332 = load i32, ptr %238, align 4, !tbaa !2
  %333 = load i32, ptr %268, align 4, !tbaa !2
  %334 = xor i32 %333, %332
  store i32 %334, ptr %240, align 4, !tbaa !2
  %335 = getelementptr inbounds nuw i8, ptr %.pn133408, i64 8
  %336 = load i32, ptr %335, align 4, !tbaa !2
  %337 = load i32, ptr %163, align 64, !tbaa !2
  %338 = add i32 %337, %331
  store i32 %338, ptr %183, align 64, !tbaa !2
  %339 = load i32, ptr %238, align 4, !tbaa !2
  %340 = load i32, ptr %240, align 4, !tbaa !2
  %341 = add i32 %340, %339
  store i32 %341, ptr %241, align 4, !tbaa !2
  %342 = load i32, ptr %335, align 4, !tbaa !2
  store i32 %342, ptr %266, align 64, !tbaa !2
  store i32 %342, ptr %268, align 4, !tbaa !2
  %343 = load i32, ptr %178, align 64, !tbaa !2
  %344 = shl i32 %343, %342
  %345 = icmp ult i32 %342, 32
  %346 = select i1 %345, i32 %344, i32 0
  store i32 %346, ptr %188, align 64, !tbaa !2
  %347 = load i32, ptr %240, align 4, !tbaa !2
  %348 = load i32, ptr %268, align 4, !tbaa !2
  %349 = shl i32 %347, %348
  %350 = icmp ult i32 %348, 32
  %351 = select i1 %350, i32 %349, i32 0
  store i32 %351, ptr %242, align 4, !tbaa !2
  %352 = sub i32 32, %336
  store i32 %352, ptr %193, align 64, !tbaa !2
  store i32 %352, ptr %266, align 64, !tbaa !2
  %353 = load i32, ptr %193, align 64, !tbaa !2
  store i32 %353, ptr %268, align 4, !tbaa !2
  %354 = load i32, ptr %178, align 64, !tbaa !2
  %355 = lshr i32 %354, %352
  %356 = icmp ult i32 %352, 32
  %357 = select i1 %356, i32 %355, i32 0
  store i32 %357, ptr %266, align 64, !tbaa !2
  %358 = load i32, ptr %240, align 4, !tbaa !2
  %359 = lshr i32 %358, %353
  %360 = icmp ult i32 %353, 32
  %361 = select i1 %360, i32 %359, i32 0
  store i32 %361, ptr %268, align 4, !tbaa !2
  %362 = load i32, ptr %188, align 64, !tbaa !2
  %363 = or i32 %362, %357
  store i32 %363, ptr %266, align 64, !tbaa !2
  %364 = load i32, ptr %242, align 4, !tbaa !2
  %365 = or i32 %364, %361
  store i32 %365, ptr %268, align 4, !tbaa !2
  %366 = load i32, ptr %183, align 64, !tbaa !2
  %367 = xor i32 %366, %363
  store i32 %367, ptr %198, align 64, !tbaa !2
  %368 = load i32, ptr %241, align 4, !tbaa !2
  %369 = load i32, ptr %268, align 4, !tbaa !2
  %370 = xor i32 %369, %368
  store i32 %370, ptr %243, align 4, !tbaa !2
  %371 = getelementptr inbounds nuw i8, ptr %.pn133408, i64 12
  %372 = load i32, ptr %371, align 4, !tbaa !2
  %373 = load i32, ptr %183, align 64, !tbaa !2
  %374 = add i32 %373, %367
  store i32 %374, ptr %203, align 64, !tbaa !2
  %375 = load i32, ptr %241, align 4, !tbaa !2
  %376 = load i32, ptr %243, align 4, !tbaa !2
  %377 = add i32 %376, %375
  store i32 %377, ptr %244, align 4, !tbaa !2
  %378 = load i32, ptr %371, align 4, !tbaa !2
  store i32 %378, ptr %266, align 64, !tbaa !2
  store i32 %378, ptr %268, align 4, !tbaa !2
  %379 = load i32, ptr %198, align 64, !tbaa !2
  %380 = shl i32 %379, %378
  %381 = icmp ult i32 %378, 32
  %382 = select i1 %381, i32 %380, i32 0
  store i32 %382, ptr %208, align 64, !tbaa !2
  %383 = load i32, ptr %243, align 4, !tbaa !2
  %384 = load i32, ptr %268, align 4, !tbaa !2
  %385 = shl i32 %383, %384
  %386 = icmp ult i32 %384, 32
  %387 = select i1 %386, i32 %385, i32 0
  store i32 %387, ptr %245, align 4, !tbaa !2
  %388 = sub i32 32, %372
  store i32 %388, ptr %213, align 64, !tbaa !2
  store i32 %388, ptr %266, align 64, !tbaa !2
  %389 = load i32, ptr %213, align 64, !tbaa !2
  store i32 %389, ptr %268, align 4, !tbaa !2
  %390 = load i32, ptr %198, align 64, !tbaa !2
  %391 = lshr i32 %390, %388
  %392 = icmp ult i32 %388, 32
  %393 = select i1 %392, i32 %391, i32 0
  store i32 %393, ptr %266, align 64, !tbaa !2
  %394 = load i32, ptr %243, align 4, !tbaa !2
  %395 = lshr i32 %394, %389
  %396 = icmp ult i32 %389, 32
  %397 = select i1 %396, i32 %395, i32 0
  store i32 %397, ptr %268, align 4, !tbaa !2
  %398 = load i32, ptr %208, align 64, !tbaa !2
  %399 = or i32 %398, %393
  store i32 %399, ptr %266, align 64, !tbaa !2
  %400 = load i32, ptr %245, align 4, !tbaa !2
  %401 = or i32 %400, %397
  store i32 %401, ptr %268, align 4, !tbaa !2
  %402 = load i32, ptr %203, align 64, !tbaa !2
  %403 = xor i32 %402, %399
  store i32 %403, ptr %218, align 64, !tbaa !2
  %404 = load i32, ptr %244, align 4, !tbaa !2
  %405 = load i32, ptr %268, align 4, !tbaa !2
  %406 = xor i32 %405, %404
  store i32 %406, ptr %246, align 4, !tbaa !2
  %407 = load i32, ptr %133, align 64, !tbaa !2
  store i32 %407, ptr %266, align 64, !tbaa !2
  store i32 %407, ptr %268, align 4, !tbaa !2
  %408 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %409 = ptrtoint ptr %408 to i64
  %410 = add i64 %409, 63
  %411 = and i64 %410, -64
  %412 = inttoptr i64 %411 to ptr
  %413 = load i32, ptr %203, align 64, !tbaa !2
  %414 = load i32, ptr %266, align 64, !tbaa !2
  %415 = add i32 %414, %413
  store i32 %415, ptr %412, align 64, !tbaa !2
  %416 = load i32, ptr %244, align 4, !tbaa !2
  %417 = load i32, ptr %268, align 4, !tbaa !2
  %418 = add i32 %417, %416
  %419 = getelementptr inbounds nuw i8, ptr %412, i64 4
  store i32 %418, ptr %419, align 4, !tbaa !2
  %420 = load i32, ptr %138, align 64, !tbaa !2
  store i32 %420, ptr %266, align 64, !tbaa !2
  store i32 %420, ptr %268, align 4, !tbaa !2
  %421 = load i32, ptr %218, align 64, !tbaa !2
  %422 = add i32 %421, %420
  store i32 %422, ptr %223, align 64, !tbaa !2
  %423 = load i32, ptr %246, align 4, !tbaa !2
  %424 = load i32, ptr %268, align 4, !tbaa !2
  %425 = add i32 %424, %423
  store i32 %425, ptr %247, align 4, !tbaa !2
  %426 = add nuw nsw i32 %251, 1
  store i32 %426, ptr %228, align 64, !tbaa !2
  store i32 %426, ptr %266, align 64, !tbaa !2
  %427 = load i32, ptr %228, align 64, !tbaa !2
  store i32 %427, ptr %268, align 4, !tbaa !2
  %428 = load i32, ptr %223, align 64, !tbaa !2
  %429 = add i32 %428, %426
  store i32 %429, ptr %266, align 64, !tbaa !2
  %430 = load i32, ptr %247, align 4, !tbaa !2
  %431 = add i32 %430, %427
  store i32 %431, ptr %268, align 4, !tbaa !2
  %432 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 80)
  %433 = ptrtoint ptr %432 to i64
  %434 = add i64 %433, 63
  %435 = and i64 %434, -64
  %436 = inttoptr i64 %435 to ptr
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 64 dereferenceable(16) %436, ptr noundef nonnull align 1 dereferenceable(16) %.pn143409, i64 16, i1 false)
  tail call void @_mlir_memref_to_llvm_free(ptr nonnull %.pn143409)
  %437 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 80)
  %438 = ptrtoint ptr %437 to i64
  %439 = add i64 %438, 63
  %440 = and i64 %439, -64
  %441 = inttoptr i64 %440 to ptr
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 64 dereferenceable(16) %441, ptr noundef nonnull align 1 dereferenceable(16) %.pn133408, i64 16, i1 false)
  tail call void @_mlir_memref_to_llvm_free(ptr nonnull %.pn133408)
  %442 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 8)
  %443 = load i64, ptr %266, align 64
  store i64 %443, ptr %442, align 1
  tail call void @_mlir_memref_to_llvm_free(ptr %254)
  %444 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 8)
  %445 = load i64, ptr %412, align 64
  store i64 %445, ptr %444, align 1
  tail call void @_mlir_memref_to_llvm_free(ptr %408)
  %446 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 16)
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(16) %446, ptr noundef nonnull align 64 dereferenceable(16) %436, i64 16, i1 false)
  tail call void @_mlir_memref_to_llvm_free(ptr %432)
  %447 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 16)
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(16) %447, ptr noundef nonnull align 64 dereferenceable(16) %441, i64 16, i1 false)
  tail call void @_mlir_memref_to_llvm_free(ptr %437)
  %448 = add nuw nsw i64 %252, 1
  %exitcond.not = icmp eq i64 %448, 5
  br i1 %exitcond.not, label %.preheader370, label %.preheader392.preheader

.preheader370:                                    ; preds = %.preheader392.preheader
  tail call void @_mlir_memref_to_llvm_free(ptr nonnull %447)
  tail call void @_mlir_memref_to_llvm_free(ptr nonnull %446)
  tail call void @_mlir_memref_to_llvm_free(ptr %224)
  tail call void @_mlir_memref_to_llvm_free(ptr %219)
  tail call void @_mlir_memref_to_llvm_free(ptr %214)
  tail call void @_mlir_memref_to_llvm_free(ptr %209)
  tail call void @_mlir_memref_to_llvm_free(ptr %204)
  tail call void @_mlir_memref_to_llvm_free(ptr %199)
  tail call void @_mlir_memref_to_llvm_free(ptr %194)
  tail call void @_mlir_memref_to_llvm_free(ptr %189)
  tail call void @_mlir_memref_to_llvm_free(ptr %184)
  tail call void @_mlir_memref_to_llvm_free(ptr %179)
  tail call void @_mlir_memref_to_llvm_free(ptr %174)
  tail call void @_mlir_memref_to_llvm_free(ptr %169)
  tail call void @_mlir_memref_to_llvm_free(ptr %164)
  tail call void @_mlir_memref_to_llvm_free(ptr %159)
  tail call void @_mlir_memref_to_llvm_free(ptr %154)
  tail call void @_mlir_memref_to_llvm_free(ptr %149)
  tail call void @_mlir_memref_to_llvm_free(ptr %144)
  tail call void @_mlir_memref_to_llvm_free(ptr %139)
  tail call void @_mlir_memref_to_llvm_free(ptr %134)
  tail call void @_mlir_memref_to_llvm_free(ptr %129)
  %449 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 80)
  %450 = ptrtoint ptr %449 to i64
  %451 = add i64 %450, 63
  %452 = and i64 %451, -64
  %453 = inttoptr i64 %452 to ptr
  %454 = load i32, ptr %444, align 4, !tbaa !2
  store i32 %454, ptr %453, align 64, !tbaa !2
  %455 = load i32, ptr %442, align 4, !tbaa !2
  %456 = getelementptr i8, ptr %453, i64 4
  store i32 %455, ptr %456, align 4, !tbaa !2
  %457 = getelementptr i8, ptr %453, i64 8
  %.in325.1445 = getelementptr inbounds nuw i8, ptr %444, i64 4
  %458 = load i32, ptr %.in325.1445, align 4, !tbaa !2
  store i32 %458, ptr %457, align 8, !tbaa !2
  %.in325.1.1 = getelementptr inbounds nuw i8, ptr %442, i64 4
  %459 = load i32, ptr %.in325.1.1, align 4, !tbaa !2
  %460 = getelementptr i8, ptr %453, i64 12
  store i32 %459, ptr %460, align 4, !tbaa !2
  tail call void @_mlir_memref_to_llvm_free(ptr nonnull %444)
  tail call void @_mlir_memref_to_llvm_free(ptr nonnull %442)
  %461 = load i32, ptr %457, align 8, !tbaa !2
  %462 = load i32, ptr %460, align 4, !tbaa !2
  %463 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 80)
  %464 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 80)
  %465 = ptrtoint ptr %464 to i64
  %466 = add i64 %465, 63
  %467 = and i64 %466, -64
  %468 = inttoptr i64 %467 to ptr
  store i64 0, ptr %468, align 64, !tbaa !2
  %469 = getelementptr inbounds nuw i8, ptr %468, i64 8
  store i64 1, ptr %469, align 8, !tbaa !2
  %470 = ptrtoint ptr %463 to i64
  %471 = add i64 %470, 63
  %472 = and i64 %471, -64
  %473 = inttoptr i64 %472 to ptr
  store i64 1, ptr %473, align 64, !tbaa !2
  %474 = getelementptr inbounds nuw i8, ptr %473, i64 8
  store i64 1, ptr %474, align 8, !tbaa !2
  %475 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 80)
  %476 = ptrtoint ptr %475 to i64
  %477 = add i64 %476, 63
  %478 = and i64 %477, -64
  %479 = inttoptr i64 %478 to ptr
  %480 = load i64, ptr %473, align 64, !tbaa !2
  %481 = load i64, ptr %468, align 64, !tbaa !2
  %482 = mul i64 %481, %480
  store i64 %482, ptr %479, align 64, !tbaa !2
  %483 = load i64, ptr %474, align 8, !tbaa !2
  %484 = load i64, ptr %469, align 8, !tbaa !2
  %485 = mul i64 %484, %483
  %486 = getelementptr inbounds nuw i8, ptr %479, i64 8
  store i64 %485, ptr %486, align 8, !tbaa !2
  tail call void @_mlir_memref_to_llvm_free(ptr %464)
  store i64 32, ptr %473, align 64, !tbaa !2
  store i64 32, ptr %474, align 8, !tbaa !2
  %487 = load i64, ptr %479, align 64, !tbaa !2
  %488 = lshr i64 %487, 32
  store i64 %488, ptr %473, align 64, !tbaa !2
  %489 = load i64, ptr %486, align 8, !tbaa !2
  %490 = lshr i64 %489, 32
  store i64 %490, ptr %474, align 8, !tbaa !2
  %491 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %492 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %493 = ptrtoint ptr %492 to i64
  %494 = add i64 %493, 63
  %495 = and i64 %494, -64
  %496 = inttoptr i64 %495 to ptr
  %497 = load i64, ptr %479, align 64, !tbaa !2
  %498 = trunc i64 %497 to i32
  store i32 %498, ptr %496, align 64, !tbaa !2
  %499 = load i64, ptr %486, align 8, !tbaa !2
  %500 = trunc i64 %499 to i32
  %501 = getelementptr inbounds nuw i8, ptr %496, i64 4
  store i32 %500, ptr %501, align 4, !tbaa !2
  tail call void @_mlir_memref_to_llvm_free(ptr %475)
  %502 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %503 = ptrtoint ptr %502 to i64
  %504 = add i64 %503, 63
  %505 = and i64 %504, -64
  %506 = inttoptr i64 %505 to ptr
  %507 = load i64, ptr %473, align 64, !tbaa !2
  %508 = trunc i64 %507 to i32
  store i32 %508, ptr %506, align 64, !tbaa !2
  %509 = load i64, ptr %474, align 8, !tbaa !2
  %510 = trunc i64 %509 to i32
  %511 = getelementptr inbounds nuw i8, ptr %506, i64 4
  store i32 %510, ptr %511, align 4, !tbaa !2
  %512 = ptrtoint ptr %491 to i64
  %513 = add i64 %512, 63
  %514 = and i64 %513, -64
  %515 = inttoptr i64 %514 to ptr
  tail call void @_mlir_memref_to_llvm_free(ptr %463)
  %516 = load i32, ptr %457, align 8, !tbaa !2
  store i32 %516, ptr %515, align 64, !tbaa !2
  %517 = getelementptr inbounds nuw i8, ptr %515, i64 4
  store i32 %516, ptr %517, align 4, !tbaa !2
  %518 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %519 = ptrtoint ptr %518 to i64
  %520 = add i64 %519, 63
  %521 = and i64 %520, -64
  %522 = inttoptr i64 %521 to ptr
  %523 = load i32, ptr %506, align 64, !tbaa !2
  %524 = load i32, ptr %515, align 64, !tbaa !2
  %525 = add i32 %524, %523
  store i32 %525, ptr %522, align 64, !tbaa !2
  %526 = load i32, ptr %511, align 4, !tbaa !2
  %527 = load i32, ptr %517, align 4, !tbaa !2
  %528 = add i32 %527, %526
  %529 = getelementptr inbounds nuw i8, ptr %522, i64 4
  store i32 %528, ptr %529, align 4, !tbaa !2
  tail call void @_mlir_memref_to_llvm_free(ptr %502)
  %530 = load i32, ptr %460, align 4, !tbaa !2
  store i32 %530, ptr %515, align 64, !tbaa !2
  store i32 %530, ptr %517, align 4, !tbaa !2
  %531 = load i32, ptr %496, align 64, !tbaa !2
  %532 = add i32 %531, %530
  store i32 %532, ptr %515, align 64, !tbaa !2
  %533 = load i32, ptr %501, align 4, !tbaa !2
  %534 = add i32 %533, %530
  store i32 %534, ptr %517, align 4, !tbaa !2
  %535 = xor i32 %461, %462
  %536 = xor i32 %535, 466688986
  tail call void @_mlir_memref_to_llvm_free(ptr %492)
  %537 = load i32, ptr %460, align 4, !tbaa !2
  %538 = load i32, ptr %457, align 8, !tbaa !2
  %539 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 80)
  %540 = ptrtoint ptr %539 to i64
  %541 = add i64 %540, 63
  %542 = and i64 %541, -64
  %543 = inttoptr i64 %542 to ptr
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 64 dereferenceable(16) %543, ptr noundef nonnull align 64 dereferenceable(16) @__constant_4xi32_3, i64 16, i1 false)
  %544 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 80)
  %545 = ptrtoint ptr %544 to i64
  %546 = add i64 %545, 63
  %547 = and i64 %546, -64
  %548 = inttoptr i64 %547 to ptr
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 64 dereferenceable(16) %548, ptr noundef nonnull align 64 dereferenceable(16) @__constant_4xi32, i64 16, i1 false)
  %549 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 68)
  %550 = ptrtoint ptr %549 to i64
  %551 = add i64 %550, 63
  %552 = and i64 %551, -64
  %553 = inttoptr i64 %552 to ptr
  %554 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 68)
  %555 = ptrtoint ptr %554 to i64
  %556 = add i64 %555, 63
  %557 = and i64 %556, -64
  %558 = inttoptr i64 %557 to ptr
  %559 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %560 = ptrtoint ptr %559 to i64
  %561 = add i64 %560, 63
  %562 = and i64 %561, -64
  %563 = inttoptr i64 %562 to ptr
  %564 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %565 = ptrtoint ptr %564 to i64
  %566 = add i64 %565, 63
  %567 = and i64 %566, -64
  %568 = inttoptr i64 %567 to ptr
  %569 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 68)
  %570 = ptrtoint ptr %569 to i64
  %571 = add i64 %570, 63
  %572 = and i64 %571, -64
  %573 = inttoptr i64 %572 to ptr
  %574 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %575 = ptrtoint ptr %574 to i64
  %576 = add i64 %575, 63
  %577 = and i64 %576, -64
  %578 = inttoptr i64 %577 to ptr
  %579 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %580 = ptrtoint ptr %579 to i64
  %581 = add i64 %580, 63
  %582 = and i64 %581, -64
  %583 = inttoptr i64 %582 to ptr
  %584 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %585 = ptrtoint ptr %584 to i64
  %586 = add i64 %585, 63
  %587 = and i64 %586, -64
  %588 = inttoptr i64 %587 to ptr
  %589 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 68)
  %590 = ptrtoint ptr %589 to i64
  %591 = add i64 %590, 63
  %592 = and i64 %591, -64
  %593 = inttoptr i64 %592 to ptr
  %594 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %595 = ptrtoint ptr %594 to i64
  %596 = add i64 %595, 63
  %597 = and i64 %596, -64
  %598 = inttoptr i64 %597 to ptr
  %599 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %600 = ptrtoint ptr %599 to i64
  %601 = add i64 %600, 63
  %602 = and i64 %601, -64
  %603 = inttoptr i64 %602 to ptr
  %604 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %605 = ptrtoint ptr %604 to i64
  %606 = add i64 %605, 63
  %607 = and i64 %606, -64
  %608 = inttoptr i64 %607 to ptr
  %609 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 68)
  %610 = ptrtoint ptr %609 to i64
  %611 = add i64 %610, 63
  %612 = and i64 %611, -64
  %613 = inttoptr i64 %612 to ptr
  %614 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %615 = ptrtoint ptr %614 to i64
  %616 = add i64 %615, 63
  %617 = and i64 %616, -64
  %618 = inttoptr i64 %617 to ptr
  %619 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %620 = ptrtoint ptr %619 to i64
  %621 = add i64 %620, 63
  %622 = and i64 %621, -64
  %623 = inttoptr i64 %622 to ptr
  %624 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %625 = ptrtoint ptr %624 to i64
  %626 = add i64 %625, 63
  %627 = and i64 %626, -64
  %628 = inttoptr i64 %627 to ptr
  %629 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 68)
  %630 = ptrtoint ptr %629 to i64
  %631 = add i64 %630, 63
  %632 = and i64 %631, -64
  %633 = inttoptr i64 %632 to ptr
  %634 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %635 = ptrtoint ptr %634 to i64
  %636 = add i64 %635, 63
  %637 = and i64 %636, -64
  %638 = inttoptr i64 %637 to ptr
  %639 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %640 = ptrtoint ptr %639 to i64
  %641 = add i64 %640, 63
  %642 = and i64 %641, -64
  %643 = inttoptr i64 %642 to ptr
  %644 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 68)
  %645 = ptrtoint ptr %644 to i64
  %646 = add i64 %645, 63
  %647 = and i64 %646, -64
  %648 = inttoptr i64 %647 to ptr
  %649 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 8)
  %650 = load i64, ptr %515, align 64
  store i64 %650, ptr %649, align 1
  tail call void @_mlir_memref_to_llvm_free(ptr %491)
  %651 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 8)
  %652 = load i64, ptr %522, align 64
  store i64 %652, ptr %651, align 1
  tail call void @_mlir_memref_to_llvm_free(ptr %518)
  %653 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 16)
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(16) %653, ptr noundef nonnull align 64 dereferenceable(16) %543, i64 16, i1 false)
  tail call void @_mlir_memref_to_llvm_free(ptr %539)
  %654 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 16)
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(16) %654, ptr noundef nonnull align 64 dereferenceable(16) %548, i64 16, i1 false)
  tail call void @_mlir_memref_to_llvm_free(ptr %544)
  %655 = getelementptr inbounds nuw i8, ptr %563, i64 4
  %656 = getelementptr inbounds nuw i8, ptr %568, i64 4
  %657 = getelementptr inbounds nuw i8, ptr %578, i64 4
  %658 = getelementptr inbounds nuw i8, ptr %583, i64 4
  %659 = getelementptr inbounds nuw i8, ptr %588, i64 4
  %660 = getelementptr inbounds nuw i8, ptr %598, i64 4
  %661 = getelementptr inbounds nuw i8, ptr %603, i64 4
  %662 = getelementptr inbounds nuw i8, ptr %608, i64 4
  %663 = getelementptr inbounds nuw i8, ptr %618, i64 4
  %664 = getelementptr inbounds nuw i8, ptr %623, i64 4
  %665 = getelementptr inbounds nuw i8, ptr %628, i64 4
  %666 = getelementptr inbounds nuw i8, ptr %638, i64 4
  %667 = getelementptr inbounds nuw i8, ptr %643, i64 4
  br label %.preheader366.preheader

.preheader366.preheader:                          ; preds = %.preheader370, %.preheader366.preheader
  %.pn183413 = phi ptr [ %654, %.preheader370 ], [ %867, %.preheader366.preheader ]
  %.pn173412 = phi ptr [ %653, %.preheader370 ], [ %866, %.preheader366.preheader ]
  %668 = phi i32 [ %538, %.preheader370 ], [ %670, %.preheader366.preheader ]
  %669 = phi i32 [ %536, %.preheader370 ], [ %668, %.preheader366.preheader ]
  %670 = phi i32 [ %537, %.preheader370 ], [ %669, %.preheader366.preheader ]
  %.pn163411 = phi ptr [ %649, %.preheader370 ], [ %862, %.preheader366.preheader ]
  %.pn153410 = phi ptr [ %651, %.preheader370 ], [ %864, %.preheader366.preheader ]
  %671 = phi i32 [ 0, %.preheader370 ], [ %846, %.preheader366.preheader ]
  %672 = phi i64 [ 0, %.preheader370 ], [ %868, %.preheader366.preheader ]
  store i32 %670, ptr %553, align 64, !tbaa !2
  store i32 %669, ptr %558, align 64, !tbaa !2
  %673 = load i32, ptr %.pn173412, align 4, !tbaa !2
  %674 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %675 = load i32, ptr %.pn153410, align 4, !tbaa !2
  %676 = load i32, ptr %.pn163411, align 4, !tbaa !2
  %677 = add i32 %676, %675
  store i32 %677, ptr %563, align 64, !tbaa !2
  %678 = getelementptr inbounds nuw i8, ptr %.pn153410, i64 4
  %679 = load i32, ptr %678, align 4, !tbaa !2
  %680 = getelementptr inbounds nuw i8, ptr %.pn163411, i64 4
  %681 = load i32, ptr %680, align 4, !tbaa !2
  %682 = add i32 %681, %679
  store i32 %682, ptr %655, align 4, !tbaa !2
  %683 = ptrtoint ptr %674 to i64
  %684 = add i64 %683, 63
  %685 = and i64 %684, -64
  %686 = inttoptr i64 %685 to ptr
  tail call void @_mlir_memref_to_llvm_free(ptr nonnull %.pn153410)
  %687 = load i32, ptr %.pn173412, align 4, !tbaa !2
  store i32 %687, ptr %686, align 64, !tbaa !2
  %688 = getelementptr inbounds nuw i8, ptr %686, i64 4
  store i32 %687, ptr %688, align 4, !tbaa !2
  %689 = load i32, ptr %.pn163411, align 4, !tbaa !2
  %690 = shl i32 %689, %687
  %691 = icmp ult i32 %687, 32
  %692 = select i1 %691, i32 %690, i32 0
  store i32 %692, ptr %568, align 64, !tbaa !2
  %693 = load i32, ptr %680, align 4, !tbaa !2
  %694 = load i32, ptr %688, align 4, !tbaa !2
  %695 = shl i32 %693, %694
  %696 = icmp ult i32 %694, 32
  %697 = select i1 %696, i32 %695, i32 0
  store i32 %697, ptr %656, align 4, !tbaa !2
  %698 = sub i32 32, %673
  store i32 %698, ptr %573, align 64, !tbaa !2
  store i32 %698, ptr %686, align 64, !tbaa !2
  %699 = load i32, ptr %573, align 64, !tbaa !2
  store i32 %699, ptr %688, align 4, !tbaa !2
  %700 = load i32, ptr %.pn163411, align 4, !tbaa !2
  %701 = lshr i32 %700, %698
  %702 = icmp ult i32 %698, 32
  %703 = select i1 %702, i32 %701, i32 0
  store i32 %703, ptr %686, align 64, !tbaa !2
  %704 = load i32, ptr %680, align 4, !tbaa !2
  %705 = lshr i32 %704, %699
  %706 = icmp ult i32 %699, 32
  %707 = select i1 %706, i32 %705, i32 0
  store i32 %707, ptr %688, align 4, !tbaa !2
  tail call void @_mlir_memref_to_llvm_free(ptr nonnull %.pn163411)
  %708 = load i32, ptr %568, align 64, !tbaa !2
  %709 = load i32, ptr %686, align 64, !tbaa !2
  %710 = or i32 %709, %708
  store i32 %710, ptr %686, align 64, !tbaa !2
  %711 = load i32, ptr %656, align 4, !tbaa !2
  %712 = load i32, ptr %688, align 4, !tbaa !2
  %713 = or i32 %712, %711
  store i32 %713, ptr %688, align 4, !tbaa !2
  %714 = load i32, ptr %563, align 64, !tbaa !2
  %715 = xor i32 %714, %710
  store i32 %715, ptr %578, align 64, !tbaa !2
  %716 = load i32, ptr %655, align 4, !tbaa !2
  %717 = load i32, ptr %688, align 4, !tbaa !2
  %718 = xor i32 %717, %716
  store i32 %718, ptr %657, align 4, !tbaa !2
  %719 = getelementptr inbounds nuw i8, ptr %.pn173412, i64 4
  %720 = load i32, ptr %719, align 4, !tbaa !2
  %721 = load i32, ptr %563, align 64, !tbaa !2
  %722 = add i32 %721, %715
  store i32 %722, ptr %583, align 64, !tbaa !2
  %723 = load i32, ptr %655, align 4, !tbaa !2
  %724 = load i32, ptr %657, align 4, !tbaa !2
  %725 = add i32 %724, %723
  store i32 %725, ptr %658, align 4, !tbaa !2
  %726 = load i32, ptr %719, align 4, !tbaa !2
  store i32 %726, ptr %686, align 64, !tbaa !2
  store i32 %726, ptr %688, align 4, !tbaa !2
  %727 = load i32, ptr %578, align 64, !tbaa !2
  %728 = shl i32 %727, %726
  %729 = icmp ult i32 %726, 32
  %730 = select i1 %729, i32 %728, i32 0
  store i32 %730, ptr %588, align 64, !tbaa !2
  %731 = load i32, ptr %657, align 4, !tbaa !2
  %732 = load i32, ptr %688, align 4, !tbaa !2
  %733 = shl i32 %731, %732
  %734 = icmp ult i32 %732, 32
  %735 = select i1 %734, i32 %733, i32 0
  store i32 %735, ptr %659, align 4, !tbaa !2
  %736 = sub i32 32, %720
  store i32 %736, ptr %593, align 64, !tbaa !2
  store i32 %736, ptr %686, align 64, !tbaa !2
  %737 = load i32, ptr %593, align 64, !tbaa !2
  store i32 %737, ptr %688, align 4, !tbaa !2
  %738 = load i32, ptr %578, align 64, !tbaa !2
  %739 = lshr i32 %738, %736
  %740 = icmp ult i32 %736, 32
  %741 = select i1 %740, i32 %739, i32 0
  store i32 %741, ptr %686, align 64, !tbaa !2
  %742 = load i32, ptr %657, align 4, !tbaa !2
  %743 = lshr i32 %742, %737
  %744 = icmp ult i32 %737, 32
  %745 = select i1 %744, i32 %743, i32 0
  store i32 %745, ptr %688, align 4, !tbaa !2
  %746 = load i32, ptr %588, align 64, !tbaa !2
  %747 = or i32 %746, %741
  store i32 %747, ptr %686, align 64, !tbaa !2
  %748 = load i32, ptr %659, align 4, !tbaa !2
  %749 = or i32 %748, %745
  store i32 %749, ptr %688, align 4, !tbaa !2
  %750 = load i32, ptr %583, align 64, !tbaa !2
  %751 = xor i32 %750, %747
  store i32 %751, ptr %598, align 64, !tbaa !2
  %752 = load i32, ptr %658, align 4, !tbaa !2
  %753 = load i32, ptr %688, align 4, !tbaa !2
  %754 = xor i32 %753, %752
  store i32 %754, ptr %660, align 4, !tbaa !2
  %755 = getelementptr inbounds nuw i8, ptr %.pn173412, i64 8
  %756 = load i32, ptr %755, align 4, !tbaa !2
  %757 = load i32, ptr %583, align 64, !tbaa !2
  %758 = add i32 %757, %751
  store i32 %758, ptr %603, align 64, !tbaa !2
  %759 = load i32, ptr %658, align 4, !tbaa !2
  %760 = load i32, ptr %660, align 4, !tbaa !2
  %761 = add i32 %760, %759
  store i32 %761, ptr %661, align 4, !tbaa !2
  %762 = load i32, ptr %755, align 4, !tbaa !2
  store i32 %762, ptr %686, align 64, !tbaa !2
  store i32 %762, ptr %688, align 4, !tbaa !2
  %763 = load i32, ptr %598, align 64, !tbaa !2
  %764 = shl i32 %763, %762
  %765 = icmp ult i32 %762, 32
  %766 = select i1 %765, i32 %764, i32 0
  store i32 %766, ptr %608, align 64, !tbaa !2
  %767 = load i32, ptr %660, align 4, !tbaa !2
  %768 = load i32, ptr %688, align 4, !tbaa !2
  %769 = shl i32 %767, %768
  %770 = icmp ult i32 %768, 32
  %771 = select i1 %770, i32 %769, i32 0
  store i32 %771, ptr %662, align 4, !tbaa !2
  %772 = sub i32 32, %756
  store i32 %772, ptr %613, align 64, !tbaa !2
  store i32 %772, ptr %686, align 64, !tbaa !2
  %773 = load i32, ptr %613, align 64, !tbaa !2
  store i32 %773, ptr %688, align 4, !tbaa !2
  %774 = load i32, ptr %598, align 64, !tbaa !2
  %775 = lshr i32 %774, %772
  %776 = icmp ult i32 %772, 32
  %777 = select i1 %776, i32 %775, i32 0
  store i32 %777, ptr %686, align 64, !tbaa !2
  %778 = load i32, ptr %660, align 4, !tbaa !2
  %779 = lshr i32 %778, %773
  %780 = icmp ult i32 %773, 32
  %781 = select i1 %780, i32 %779, i32 0
  store i32 %781, ptr %688, align 4, !tbaa !2
  %782 = load i32, ptr %608, align 64, !tbaa !2
  %783 = or i32 %782, %777
  store i32 %783, ptr %686, align 64, !tbaa !2
  %784 = load i32, ptr %662, align 4, !tbaa !2
  %785 = or i32 %784, %781
  store i32 %785, ptr %688, align 4, !tbaa !2
  %786 = load i32, ptr %603, align 64, !tbaa !2
  %787 = xor i32 %786, %783
  store i32 %787, ptr %618, align 64, !tbaa !2
  %788 = load i32, ptr %661, align 4, !tbaa !2
  %789 = load i32, ptr %688, align 4, !tbaa !2
  %790 = xor i32 %789, %788
  store i32 %790, ptr %663, align 4, !tbaa !2
  %791 = getelementptr inbounds nuw i8, ptr %.pn173412, i64 12
  %792 = load i32, ptr %791, align 4, !tbaa !2
  %793 = load i32, ptr %603, align 64, !tbaa !2
  %794 = add i32 %793, %787
  store i32 %794, ptr %623, align 64, !tbaa !2
  %795 = load i32, ptr %661, align 4, !tbaa !2
  %796 = load i32, ptr %663, align 4, !tbaa !2
  %797 = add i32 %796, %795
  store i32 %797, ptr %664, align 4, !tbaa !2
  %798 = load i32, ptr %791, align 4, !tbaa !2
  store i32 %798, ptr %686, align 64, !tbaa !2
  store i32 %798, ptr %688, align 4, !tbaa !2
  %799 = load i32, ptr %618, align 64, !tbaa !2
  %800 = shl i32 %799, %798
  %801 = icmp ult i32 %798, 32
  %802 = select i1 %801, i32 %800, i32 0
  store i32 %802, ptr %628, align 64, !tbaa !2
  %803 = load i32, ptr %663, align 4, !tbaa !2
  %804 = load i32, ptr %688, align 4, !tbaa !2
  %805 = shl i32 %803, %804
  %806 = icmp ult i32 %804, 32
  %807 = select i1 %806, i32 %805, i32 0
  store i32 %807, ptr %665, align 4, !tbaa !2
  %808 = sub i32 32, %792
  store i32 %808, ptr %633, align 64, !tbaa !2
  store i32 %808, ptr %686, align 64, !tbaa !2
  %809 = load i32, ptr %633, align 64, !tbaa !2
  store i32 %809, ptr %688, align 4, !tbaa !2
  %810 = load i32, ptr %618, align 64, !tbaa !2
  %811 = lshr i32 %810, %808
  %812 = icmp ult i32 %808, 32
  %813 = select i1 %812, i32 %811, i32 0
  store i32 %813, ptr %686, align 64, !tbaa !2
  %814 = load i32, ptr %663, align 4, !tbaa !2
  %815 = lshr i32 %814, %809
  %816 = icmp ult i32 %809, 32
  %817 = select i1 %816, i32 %815, i32 0
  store i32 %817, ptr %688, align 4, !tbaa !2
  %818 = load i32, ptr %628, align 64, !tbaa !2
  %819 = or i32 %818, %813
  store i32 %819, ptr %686, align 64, !tbaa !2
  %820 = load i32, ptr %665, align 4, !tbaa !2
  %821 = or i32 %820, %817
  store i32 %821, ptr %688, align 4, !tbaa !2
  %822 = load i32, ptr %623, align 64, !tbaa !2
  %823 = xor i32 %822, %819
  store i32 %823, ptr %638, align 64, !tbaa !2
  %824 = load i32, ptr %664, align 4, !tbaa !2
  %825 = load i32, ptr %688, align 4, !tbaa !2
  %826 = xor i32 %825, %824
  store i32 %826, ptr %666, align 4, !tbaa !2
  %827 = load i32, ptr %553, align 64, !tbaa !2
  store i32 %827, ptr %686, align 64, !tbaa !2
  store i32 %827, ptr %688, align 4, !tbaa !2
  %828 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %829 = ptrtoint ptr %828 to i64
  %830 = add i64 %829, 63
  %831 = and i64 %830, -64
  %832 = inttoptr i64 %831 to ptr
  %833 = load i32, ptr %623, align 64, !tbaa !2
  %834 = load i32, ptr %686, align 64, !tbaa !2
  %835 = add i32 %834, %833
  store i32 %835, ptr %832, align 64, !tbaa !2
  %836 = load i32, ptr %664, align 4, !tbaa !2
  %837 = load i32, ptr %688, align 4, !tbaa !2
  %838 = add i32 %837, %836
  %839 = getelementptr inbounds nuw i8, ptr %832, i64 4
  store i32 %838, ptr %839, align 4, !tbaa !2
  %840 = load i32, ptr %558, align 64, !tbaa !2
  store i32 %840, ptr %686, align 64, !tbaa !2
  store i32 %840, ptr %688, align 4, !tbaa !2
  %841 = load i32, ptr %638, align 64, !tbaa !2
  %842 = add i32 %841, %840
  store i32 %842, ptr %643, align 64, !tbaa !2
  %843 = load i32, ptr %666, align 4, !tbaa !2
  %844 = load i32, ptr %688, align 4, !tbaa !2
  %845 = add i32 %844, %843
  store i32 %845, ptr %667, align 4, !tbaa !2
  %846 = add nuw nsw i32 %671, 1
  store i32 %846, ptr %648, align 64, !tbaa !2
  store i32 %846, ptr %686, align 64, !tbaa !2
  %847 = load i32, ptr %648, align 64, !tbaa !2
  store i32 %847, ptr %688, align 4, !tbaa !2
  %848 = load i32, ptr %643, align 64, !tbaa !2
  %849 = add i32 %848, %846
  store i32 %849, ptr %686, align 64, !tbaa !2
  %850 = load i32, ptr %667, align 4, !tbaa !2
  %851 = add i32 %850, %847
  store i32 %851, ptr %688, align 4, !tbaa !2
  %852 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 80)
  %853 = ptrtoint ptr %852 to i64
  %854 = add i64 %853, 63
  %855 = and i64 %854, -64
  %856 = inttoptr i64 %855 to ptr
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 64 dereferenceable(16) %856, ptr noundef nonnull align 1 dereferenceable(16) %.pn183413, i64 16, i1 false)
  tail call void @_mlir_memref_to_llvm_free(ptr nonnull %.pn183413)
  %857 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 80)
  %858 = ptrtoint ptr %857 to i64
  %859 = add i64 %858, 63
  %860 = and i64 %859, -64
  %861 = inttoptr i64 %860 to ptr
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 64 dereferenceable(16) %861, ptr noundef nonnull align 1 dereferenceable(16) %.pn173412, i64 16, i1 false)
  tail call void @_mlir_memref_to_llvm_free(ptr nonnull %.pn173412)
  %862 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 8)
  %863 = load i64, ptr %686, align 64
  store i64 %863, ptr %862, align 1
  tail call void @_mlir_memref_to_llvm_free(ptr %674)
  %864 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 8)
  %865 = load i64, ptr %832, align 64
  store i64 %865, ptr %864, align 1
  tail call void @_mlir_memref_to_llvm_free(ptr %828)
  %866 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 16)
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(16) %866, ptr noundef nonnull align 64 dereferenceable(16) %856, i64 16, i1 false)
  tail call void @_mlir_memref_to_llvm_free(ptr %852)
  %867 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 16)
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(16) %867, ptr noundef nonnull align 64 dereferenceable(16) %861, i64 16, i1 false)
  tail call void @_mlir_memref_to_llvm_free(ptr %857)
  %868 = add nuw nsw i64 %672, 1
  %exitcond446.not = icmp eq i64 %868, 5
  br i1 %exitcond446.not, label %.preheader344, label %.preheader366.preheader

.preheader344:                                    ; preds = %.preheader366.preheader
  tail call void @_mlir_memref_to_llvm_free(ptr nonnull %867)
  tail call void @_mlir_memref_to_llvm_free(ptr nonnull %866)
  tail call void @_mlir_memref_to_llvm_free(ptr %644)
  tail call void @_mlir_memref_to_llvm_free(ptr %639)
  tail call void @_mlir_memref_to_llvm_free(ptr %634)
  tail call void @_mlir_memref_to_llvm_free(ptr %629)
  tail call void @_mlir_memref_to_llvm_free(ptr %624)
  tail call void @_mlir_memref_to_llvm_free(ptr %619)
  tail call void @_mlir_memref_to_llvm_free(ptr %614)
  tail call void @_mlir_memref_to_llvm_free(ptr %609)
  tail call void @_mlir_memref_to_llvm_free(ptr %604)
  tail call void @_mlir_memref_to_llvm_free(ptr %599)
  tail call void @_mlir_memref_to_llvm_free(ptr %594)
  tail call void @_mlir_memref_to_llvm_free(ptr %589)
  tail call void @_mlir_memref_to_llvm_free(ptr %584)
  tail call void @_mlir_memref_to_llvm_free(ptr %579)
  tail call void @_mlir_memref_to_llvm_free(ptr %574)
  tail call void @_mlir_memref_to_llvm_free(ptr %569)
  tail call void @_mlir_memref_to_llvm_free(ptr %564)
  tail call void @_mlir_memref_to_llvm_free(ptr %559)
  tail call void @_mlir_memref_to_llvm_free(ptr %554)
  tail call void @_mlir_memref_to_llvm_free(ptr %549)
  %869 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 80)
  %870 = ptrtoint ptr %869 to i64
  %871 = add i64 %870, 63
  %872 = and i64 %871, -64
  %873 = inttoptr i64 %872 to ptr
  %874 = load i32, ptr %864, align 4, !tbaa !2
  store i32 %874, ptr %873, align 64, !tbaa !2
  %875 = load i32, ptr %862, align 4, !tbaa !2
  %876 = getelementptr i8, ptr %873, i64 4
  store i32 %875, ptr %876, align 4, !tbaa !2
  %877 = getelementptr i8, ptr %873, i64 8
  %.in323.1448 = getelementptr inbounds nuw i8, ptr %864, i64 4
  %878 = load i32, ptr %.in323.1448, align 4, !tbaa !2
  store i32 %878, ptr %877, align 8, !tbaa !2
  %.in323.1.1 = getelementptr inbounds nuw i8, ptr %862, i64 4
  %879 = load i32, ptr %.in323.1.1, align 4, !tbaa !2
  %880 = getelementptr i8, ptr %873, i64 12
  store i32 %879, ptr %880, align 4, !tbaa !2
  tail call void @_mlir_memref_to_llvm_free(ptr nonnull %864)
  tail call void @_mlir_memref_to_llvm_free(ptr nonnull %862)
  %881 = load i32, ptr %873, align 64, !tbaa !2
  %882 = load i32, ptr %876, align 4, !tbaa !2
  %883 = xor i32 %881, %882
  %884 = xor i32 %883, 466688986
  %885 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 80)
  %886 = ptrtoint ptr %885 to i64
  %887 = add i64 %886, 63
  %888 = and i64 %887, -64
  %889 = inttoptr i64 %888 to ptr
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 64 dereferenceable(16) %889, ptr noundef nonnull align 64 dereferenceable(16) @__constant_4xi32_3, i64 16, i1 false)
  %890 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 80)
  %891 = ptrtoint ptr %890 to i64
  %892 = add i64 %891, 63
  %893 = and i64 %892, -64
  %894 = inttoptr i64 %893 to ptr
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 64 dereferenceable(16) %894, ptr noundef nonnull align 64 dereferenceable(16) @__constant_4xi32, i64 16, i1 false)
  %895 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 16)
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(16) %895, ptr noundef nonnull align 64 dereferenceable(16) %889, i64 16, i1 false)
  tail call void @_mlir_memref_to_llvm_free(ptr %885)
  %896 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 16)
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(16) %896, ptr noundef nonnull align 64 dereferenceable(16) %894, i64 16, i1 false)
  tail call void @_mlir_memref_to_llvm_free(ptr %890)
  br label %897

897:                                              ; preds = %.preheader344, %897
  %.pn203415 = phi ptr [ %896, %.preheader344 ], [ %967, %897 ]
  %.pn193414 = phi ptr [ %895, %.preheader344 ], [ %966, %897 ]
  %898 = phi i32 [ %881, %.preheader344 ], [ %900, %897 ]
  %899 = phi i32 [ %884, %.preheader344 ], [ %898, %897 ]
  %900 = phi i32 [ %882, %.preheader344 ], [ %899, %897 ]
  %901 = phi i32 [ %882, %.preheader344 ], [ %955, %897 ]
  %902 = phi i32 [ %881, %.preheader344 ], [ %953, %897 ]
  %903 = phi i32 [ 0, %.preheader344 ], [ %905, %897 ]
  %904 = phi i64 [ 0, %.preheader344 ], [ %968, %897 ]
  %905 = add nuw nsw i32 %903, 1
  %906 = load i32, ptr %.pn193414, align 4, !tbaa !2
  %907 = add i32 %902, %901
  %908 = shl i32 %901, %906
  %909 = icmp ult i32 %906, 32
  %910 = select i1 %909, i32 %908, i32 0
  %911 = sub i32 32, %906
  %912 = lshr i32 %901, %911
  %913 = icmp ult i32 %911, 32
  %914 = select i1 %913, i32 %912, i32 0
  %915 = or i32 %914, %910
  %916 = xor i32 %915, %907
  %917 = getelementptr inbounds nuw i8, ptr %.pn193414, i64 4
  %918 = load i32, ptr %917, align 4, !tbaa !2
  %919 = add i32 %916, %907
  %920 = shl i32 %916, %918
  %921 = icmp ult i32 %918, 32
  %922 = select i1 %921, i32 %920, i32 0
  %923 = sub i32 32, %918
  %924 = lshr i32 %916, %923
  %925 = icmp ult i32 %923, 32
  %926 = select i1 %925, i32 %924, i32 0
  %927 = or i32 %922, %926
  %928 = xor i32 %927, %919
  %929 = getelementptr inbounds nuw i8, ptr %.pn193414, i64 8
  %930 = load i32, ptr %929, align 4, !tbaa !2
  %931 = add i32 %928, %919
  %932 = shl i32 %928, %930
  %933 = icmp ult i32 %930, 32
  %934 = select i1 %933, i32 %932, i32 0
  %935 = sub i32 32, %930
  %936 = lshr i32 %928, %935
  %937 = icmp ult i32 %935, 32
  %938 = select i1 %937, i32 %936, i32 0
  %939 = or i32 %934, %938
  %940 = xor i32 %939, %931
  %941 = getelementptr inbounds nuw i8, ptr %.pn193414, i64 12
  %942 = load i32, ptr %941, align 4, !tbaa !2
  %943 = add i32 %940, %931
  %944 = shl i32 %940, %942
  %945 = icmp ult i32 %942, 32
  %946 = select i1 %945, i32 %944, i32 0
  %947 = sub i32 32, %942
  %948 = lshr i32 %940, %947
  %949 = icmp ult i32 %947, 32
  %950 = select i1 %949, i32 %948, i32 0
  %951 = or i32 %946, %950
  %952 = xor i32 %951, %943
  %953 = add i32 %943, %900
  %954 = add i32 %905, %899
  %955 = add i32 %954, %952
  %956 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 80)
  %957 = ptrtoint ptr %956 to i64
  %958 = add i64 %957, 63
  %959 = and i64 %958, -64
  %960 = inttoptr i64 %959 to ptr
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 64 dereferenceable(16) %960, ptr noundef nonnull align 1 dereferenceable(16) %.pn203415, i64 16, i1 false)
  tail call void @_mlir_memref_to_llvm_free(ptr nonnull %.pn203415)
  %961 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 80)
  %962 = ptrtoint ptr %961 to i64
  %963 = add i64 %962, 63
  %964 = and i64 %963, -64
  %965 = inttoptr i64 %964 to ptr
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 64 dereferenceable(16) %965, ptr noundef nonnull align 1 dereferenceable(16) %.pn193414, i64 16, i1 false)
  tail call void @_mlir_memref_to_llvm_free(ptr nonnull %.pn193414)
  %966 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 16)
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(16) %966, ptr noundef nonnull align 64 dereferenceable(16) %960, i64 16, i1 false)
  tail call void @_mlir_memref_to_llvm_free(ptr %956)
  %967 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 16)
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(16) %967, ptr noundef nonnull align 64 dereferenceable(16) %965, i64 16, i1 false)
  tail call void @_mlir_memref_to_llvm_free(ptr %961)
  %968 = add nuw nsw i64 %904, 1
  %exitcond449.not = icmp eq i64 %968, 5
  br i1 %exitcond449.not, label %969, label %897

969:                                              ; preds = %897
  tail call void @_mlir_memref_to_llvm_free(ptr nonnull %967)
  tail call void @_mlir_memref_to_llvm_free(ptr nonnull %966)
  %970 = load i32, ptr %877, align 8, !tbaa !2
  %971 = load i32, ptr %880, align 4, !tbaa !2
  %972 = xor i32 %970, %971
  %973 = xor i32 %972, 466688986
  tail call void @_mlir_memref_to_llvm_free(ptr %869)
  %974 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 80)
  %975 = ptrtoint ptr %974 to i64
  %976 = add i64 %975, 63
  %977 = and i64 %976, -64
  %978 = inttoptr i64 %977 to ptr
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 64 dereferenceable(16) %978, ptr noundef nonnull align 64 dereferenceable(16) @__constant_4xi32_3, i64 16, i1 false)
  %979 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 80)
  %980 = ptrtoint ptr %979 to i64
  %981 = add i64 %980, 63
  %982 = and i64 %981, -64
  %983 = inttoptr i64 %982 to ptr
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 64 dereferenceable(16) %983, ptr noundef nonnull align 64 dereferenceable(16) @__constant_4xi32, i64 16, i1 false)
  %984 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 16)
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(16) %984, ptr noundef nonnull align 64 dereferenceable(16) %978, i64 16, i1 false)
  tail call void @_mlir_memref_to_llvm_free(ptr %974)
  %985 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 16)
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(16) %985, ptr noundef nonnull align 64 dereferenceable(16) %983, i64 16, i1 false)
  tail call void @_mlir_memref_to_llvm_free(ptr %979)
  br label %986

986:                                              ; preds = %969, %986
  %.pn223417 = phi ptr [ %985, %969 ], [ %1056, %986 ]
  %.pn213416 = phi ptr [ %984, %969 ], [ %1055, %986 ]
  %987 = phi i32 [ %970, %969 ], [ %989, %986 ]
  %988 = phi i32 [ %973, %969 ], [ %987, %986 ]
  %989 = phi i32 [ %971, %969 ], [ %988, %986 ]
  %990 = phi i32 [ %971, %969 ], [ %1044, %986 ]
  %991 = phi i32 [ %970, %969 ], [ %1042, %986 ]
  %992 = phi i32 [ 0, %969 ], [ %994, %986 ]
  %993 = phi i64 [ 0, %969 ], [ %1057, %986 ]
  %994 = add nuw nsw i32 %992, 1
  %995 = load i32, ptr %.pn213416, align 4, !tbaa !2
  %996 = add i32 %991, %990
  %997 = shl i32 %990, %995
  %998 = icmp ult i32 %995, 32
  %999 = select i1 %998, i32 %997, i32 0
  %1000 = sub i32 32, %995
  %1001 = lshr i32 %990, %1000
  %1002 = icmp ult i32 %1000, 32
  %1003 = select i1 %1002, i32 %1001, i32 0
  %1004 = or i32 %1003, %999
  %1005 = xor i32 %1004, %996
  %1006 = getelementptr inbounds nuw i8, ptr %.pn213416, i64 4
  %1007 = load i32, ptr %1006, align 4, !tbaa !2
  %1008 = add i32 %1005, %996
  %1009 = shl i32 %1005, %1007
  %1010 = icmp ult i32 %1007, 32
  %1011 = select i1 %1010, i32 %1009, i32 0
  %1012 = sub i32 32, %1007
  %1013 = lshr i32 %1005, %1012
  %1014 = icmp ult i32 %1012, 32
  %1015 = select i1 %1014, i32 %1013, i32 0
  %1016 = or i32 %1011, %1015
  %1017 = xor i32 %1016, %1008
  %1018 = getelementptr inbounds nuw i8, ptr %.pn213416, i64 8
  %1019 = load i32, ptr %1018, align 4, !tbaa !2
  %1020 = add i32 %1017, %1008
  %1021 = shl i32 %1017, %1019
  %1022 = icmp ult i32 %1019, 32
  %1023 = select i1 %1022, i32 %1021, i32 0
  %1024 = sub i32 32, %1019
  %1025 = lshr i32 %1017, %1024
  %1026 = icmp ult i32 %1024, 32
  %1027 = select i1 %1026, i32 %1025, i32 0
  %1028 = or i32 %1023, %1027
  %1029 = xor i32 %1028, %1020
  %1030 = getelementptr inbounds nuw i8, ptr %.pn213416, i64 12
  %1031 = load i32, ptr %1030, align 4, !tbaa !2
  %1032 = add i32 %1029, %1020
  %1033 = shl i32 %1029, %1031
  %1034 = icmp ult i32 %1031, 32
  %1035 = select i1 %1034, i32 %1033, i32 0
  %1036 = sub i32 32, %1031
  %1037 = lshr i32 %1029, %1036
  %1038 = icmp ult i32 %1036, 32
  %1039 = select i1 %1038, i32 %1037, i32 0
  %1040 = or i32 %1035, %1039
  %1041 = xor i32 %1040, %1032
  %1042 = add i32 %1032, %989
  %1043 = add i32 %994, %988
  %1044 = add i32 %1043, %1041
  %1045 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 80)
  %1046 = ptrtoint ptr %1045 to i64
  %1047 = add i64 %1046, 63
  %1048 = and i64 %1047, -64
  %1049 = inttoptr i64 %1048 to ptr
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 64 dereferenceable(16) %1049, ptr noundef nonnull align 1 dereferenceable(16) %.pn223417, i64 16, i1 false)
  tail call void @_mlir_memref_to_llvm_free(ptr nonnull %.pn223417)
  %1050 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 80)
  %1051 = ptrtoint ptr %1050 to i64
  %1052 = add i64 %1051, 63
  %1053 = and i64 %1052, -64
  %1054 = inttoptr i64 %1053 to ptr
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 64 dereferenceable(16) %1054, ptr noundef nonnull align 1 dereferenceable(16) %.pn213416, i64 16, i1 false)
  tail call void @_mlir_memref_to_llvm_free(ptr nonnull %.pn213416)
  %1055 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 16)
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(16) %1055, ptr noundef nonnull align 64 dereferenceable(16) %1049, i64 16, i1 false)
  tail call void @_mlir_memref_to_llvm_free(ptr %1045)
  %1056 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 16)
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(16) %1056, ptr noundef nonnull align 64 dereferenceable(16) %1054, i64 16, i1 false)
  tail call void @_mlir_memref_to_llvm_free(ptr %1050)
  %1057 = add nuw nsw i64 %993, 1
  %exitcond450.not = icmp eq i64 %1057, 5
  br i1 %exitcond450.not, label %1058, label %986

1058:                                             ; preds = %986
  %1059 = zext i32 %953 to i64
  %1060 = zext i32 %955 to i64
  %1061 = shl nuw i64 %1059, 32
  %1062 = or disjoint i64 %1061, %1060
  tail call void @_mlir_memref_to_llvm_free(ptr nonnull %1056)
  tail call void @_mlir_memref_to_llvm_free(ptr nonnull %1055)
  %1063 = zext i32 %1042 to i64
  %1064 = zext i32 %1044 to i64
  %1065 = shl nuw i64 %1063, 32
  %1066 = or disjoint i64 %1065, %1064
  %1067 = urem i64 %1062, 96
  %1068 = shl nuw nsw i64 %1067, 6
  %1069 = urem i64 %1066, 96
  %1070 = add nuw nsw i64 %1069, %1068
  %.lhs.trunc = trunc nuw nsw i64 %1070 to i16
  %1071 = urem i16 %.lhs.trunc, 96
  %.zext = zext nneg i16 %1071 to i64
  %1072 = load float, ptr %10, align 4, !tbaa !5
  %1073 = load float, ptr %13, align 4, !tbaa !5
  %1074 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 68)
  %1075 = ptrtoint ptr %1074 to i64
  %1076 = add i64 %1075, 63
  %1077 = and i64 %1076, -64
  %1078 = inttoptr i64 %1077 to ptr
  %1079 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 68)
  %1080 = ptrtoint ptr %1079 to i64
  %1081 = add i64 %1080, 63
  %1082 = and i64 %1081, -64
  %1083 = inttoptr i64 %1082 to ptr
  %1084 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 448)
  %1085 = ptrtoint ptr %1084 to i64
  %1086 = add i64 %1085, 63
  %1087 = and i64 %1086, -64
  %1088 = inttoptr i64 %1087 to ptr
  %1089 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 96)
  %1090 = ptrtoint ptr %1089 to i64
  %1091 = add i64 %1090, 63
  %1092 = and i64 %1091, -64
  %1093 = inttoptr i64 %1092 to ptr
  %1094 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %1095 = ptrtoint ptr %1094 to i64
  %1096 = add i64 %1095, 63
  %1097 = and i64 %1096, -64
  %1098 = inttoptr i64 %1097 to ptr
  %1099 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 384)
  %1100 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 4)
  %1101 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 4)
  %1102 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 8)
  %1103 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 448)
  %1104 = ptrtoint ptr %1103 to i64
  %1105 = add i64 %1104, 63
  %1106 = and i64 %1105, -64
  %1107 = inttoptr i64 %1106 to ptr
  %1108 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 384)
  %1109 = shl i64 %3, 2
  %1110 = mul i64 %1109, %4
  %1111 = mul i64 %1110, %5
  %1112 = getelementptr inbounds float, ptr %1, i64 %2
  tail call void @llvm.memcpy.p0.p0.i64(ptr align 1 %1108, ptr align 1 %1112, i64 %1111, i1 false)
  %1113 = getelementptr inbounds nuw i8, ptr %1093, i64 4
  %1114 = getelementptr inbounds nuw i8, ptr %1093, i64 8
  %1115 = getelementptr inbounds nuw i8, ptr %1093, i64 12
  %1116 = getelementptr inbounds nuw i8, ptr %1093, i64 16
  %1117 = getelementptr inbounds nuw i8, ptr %1093, i64 20
  %1118 = getelementptr inbounds nuw i8, ptr %1093, i64 24
  %1119 = getelementptr inbounds nuw i8, ptr %1093, i64 28
  %1120 = getelementptr inbounds nuw i8, ptr %1099, i64 4
  %1121 = getelementptr inbounds nuw i8, ptr %1099, i64 8
  %1122 = getelementptr inbounds nuw i8, ptr %1099, i64 12
  %1123 = getelementptr inbounds nuw i8, ptr %1099, i64 16
  %1124 = getelementptr inbounds nuw i8, ptr %1099, i64 20
  %1125 = getelementptr inbounds nuw i8, ptr %1099, i64 24
  %1126 = getelementptr inbounds nuw i8, ptr %1099, i64 28
  %1127 = getelementptr inbounds nuw i8, ptr %1099, i64 32
  %1128 = getelementptr inbounds nuw i8, ptr %1099, i64 36
  %1129 = getelementptr inbounds nuw i8, ptr %1099, i64 40
  %1130 = getelementptr inbounds nuw i8, ptr %1099, i64 44
  %1131 = getelementptr inbounds nuw i8, ptr %1099, i64 48
  %1132 = getelementptr inbounds nuw i8, ptr %1099, i64 52
  %1133 = getelementptr inbounds nuw i8, ptr %1099, i64 56
  %1134 = getelementptr inbounds nuw i8, ptr %1099, i64 60
  %1135 = getelementptr inbounds nuw i8, ptr %1099, i64 64
  %1136 = getelementptr inbounds nuw i8, ptr %1099, i64 68
  %1137 = getelementptr inbounds nuw i8, ptr %1099, i64 72
  %1138 = getelementptr inbounds nuw i8, ptr %1099, i64 76
  %1139 = getelementptr inbounds nuw i8, ptr %1099, i64 80
  %1140 = getelementptr inbounds nuw i8, ptr %1099, i64 84
  %1141 = getelementptr inbounds nuw i8, ptr %1099, i64 88
  %1142 = getelementptr inbounds nuw i8, ptr %1099, i64 92
  %1143 = getelementptr inbounds nuw i8, ptr %1099, i64 96
  %1144 = getelementptr inbounds nuw i8, ptr %1099, i64 100
  %1145 = getelementptr inbounds nuw i8, ptr %1099, i64 104
  %1146 = getelementptr inbounds nuw i8, ptr %1099, i64 108
  %1147 = getelementptr inbounds nuw i8, ptr %1099, i64 112
  %1148 = getelementptr inbounds nuw i8, ptr %1099, i64 116
  %1149 = getelementptr inbounds nuw i8, ptr %1099, i64 120
  %1150 = getelementptr inbounds nuw i8, ptr %1099, i64 124
  %1151 = getelementptr inbounds nuw i8, ptr %1099, i64 128
  %1152 = getelementptr inbounds nuw i8, ptr %1099, i64 132
  %1153 = getelementptr inbounds nuw i8, ptr %1099, i64 136
  %1154 = getelementptr inbounds nuw i8, ptr %1099, i64 140
  %1155 = getelementptr inbounds nuw i8, ptr %1099, i64 144
  %1156 = getelementptr inbounds nuw i8, ptr %1099, i64 148
  %1157 = getelementptr inbounds nuw i8, ptr %1099, i64 152
  %1158 = getelementptr inbounds nuw i8, ptr %1099, i64 156
  %1159 = getelementptr inbounds nuw i8, ptr %1099, i64 160
  %1160 = getelementptr inbounds nuw i8, ptr %1099, i64 164
  %1161 = getelementptr inbounds nuw i8, ptr %1099, i64 168
  %1162 = getelementptr inbounds nuw i8, ptr %1099, i64 172
  %1163 = getelementptr inbounds nuw i8, ptr %1099, i64 176
  %1164 = getelementptr inbounds nuw i8, ptr %1099, i64 180
  %1165 = getelementptr inbounds nuw i8, ptr %1099, i64 184
  %1166 = getelementptr inbounds nuw i8, ptr %1099, i64 188
  %1167 = getelementptr inbounds nuw i8, ptr %1099, i64 192
  %1168 = getelementptr inbounds nuw i8, ptr %1099, i64 196
  %1169 = getelementptr inbounds nuw i8, ptr %1099, i64 200
  %1170 = getelementptr inbounds nuw i8, ptr %1099, i64 204
  %1171 = getelementptr inbounds nuw i8, ptr %1099, i64 208
  %1172 = getelementptr inbounds nuw i8, ptr %1099, i64 212
  %1173 = getelementptr inbounds nuw i8, ptr %1099, i64 216
  %1174 = getelementptr inbounds nuw i8, ptr %1099, i64 220
  %1175 = getelementptr inbounds nuw i8, ptr %1099, i64 224
  %1176 = getelementptr inbounds nuw i8, ptr %1099, i64 228
  %1177 = getelementptr inbounds nuw i8, ptr %1099, i64 232
  %1178 = getelementptr inbounds nuw i8, ptr %1099, i64 236
  %1179 = getelementptr inbounds nuw i8, ptr %1099, i64 240
  %1180 = getelementptr inbounds nuw i8, ptr %1099, i64 244
  %1181 = getelementptr inbounds nuw i8, ptr %1099, i64 248
  %1182 = getelementptr inbounds nuw i8, ptr %1099, i64 252
  %1183 = getelementptr inbounds nuw i8, ptr %1099, i64 256
  %1184 = getelementptr inbounds nuw i8, ptr %1099, i64 260
  %1185 = getelementptr inbounds nuw i8, ptr %1099, i64 264
  %1186 = getelementptr inbounds nuw i8, ptr %1099, i64 268
  %1187 = getelementptr inbounds nuw i8, ptr %1099, i64 272
  %1188 = getelementptr inbounds nuw i8, ptr %1099, i64 276
  %1189 = getelementptr inbounds nuw i8, ptr %1099, i64 280
  %1190 = getelementptr inbounds nuw i8, ptr %1099, i64 284
  %1191 = getelementptr inbounds nuw i8, ptr %1099, i64 288
  %1192 = getelementptr inbounds nuw i8, ptr %1099, i64 292
  %1193 = getelementptr inbounds nuw i8, ptr %1099, i64 296
  %1194 = getelementptr inbounds nuw i8, ptr %1099, i64 300
  %1195 = getelementptr inbounds nuw i8, ptr %1099, i64 304
  %1196 = getelementptr inbounds nuw i8, ptr %1099, i64 308
  %1197 = getelementptr inbounds nuw i8, ptr %1099, i64 312
  %1198 = getelementptr inbounds nuw i8, ptr %1099, i64 316
  %1199 = getelementptr inbounds nuw i8, ptr %1099, i64 320
  %1200 = getelementptr inbounds nuw i8, ptr %1099, i64 324
  %1201 = getelementptr inbounds nuw i8, ptr %1099, i64 328
  %1202 = getelementptr inbounds nuw i8, ptr %1099, i64 332
  %1203 = getelementptr inbounds nuw i8, ptr %1099, i64 336
  %1204 = getelementptr inbounds nuw i8, ptr %1099, i64 340
  %1205 = getelementptr inbounds nuw i8, ptr %1099, i64 344
  %1206 = getelementptr inbounds nuw i8, ptr %1099, i64 348
  %1207 = getelementptr inbounds nuw i8, ptr %1099, i64 352
  %1208 = getelementptr inbounds nuw i8, ptr %1099, i64 356
  %1209 = getelementptr inbounds nuw i8, ptr %1099, i64 360
  %1210 = getelementptr inbounds nuw i8, ptr %1099, i64 364
  %1211 = getelementptr inbounds nuw i8, ptr %1099, i64 368
  %1212 = getelementptr inbounds nuw i8, ptr %1099, i64 372
  %1213 = getelementptr inbounds nuw i8, ptr %1099, i64 376
  %1214 = getelementptr inbounds nuw i8, ptr %1099, i64 380
  %1215 = getelementptr inbounds nuw i8, ptr %1107, i64 4
  %1216 = getelementptr inbounds nuw i8, ptr %1107, i64 8
  %1217 = getelementptr inbounds nuw i8, ptr %1107, i64 12
  %1218 = getelementptr inbounds nuw i8, ptr %1107, i64 16
  %1219 = getelementptr inbounds nuw i8, ptr %1107, i64 20
  %1220 = getelementptr inbounds nuw i8, ptr %1107, i64 24
  %1221 = getelementptr inbounds nuw i8, ptr %1107, i64 28
  %1222 = getelementptr inbounds nuw i8, ptr %1107, i64 32
  %1223 = getelementptr inbounds nuw i8, ptr %1107, i64 36
  %1224 = getelementptr inbounds nuw i8, ptr %1107, i64 40
  %1225 = getelementptr inbounds nuw i8, ptr %1107, i64 44
  %1226 = getelementptr inbounds nuw i8, ptr %1107, i64 48
  %1227 = getelementptr inbounds nuw i8, ptr %1107, i64 52
  %1228 = getelementptr inbounds nuw i8, ptr %1107, i64 56
  %1229 = getelementptr inbounds nuw i8, ptr %1107, i64 60
  %1230 = getelementptr inbounds nuw i8, ptr %1107, i64 64
  %1231 = getelementptr inbounds nuw i8, ptr %1107, i64 68
  %1232 = getelementptr inbounds nuw i8, ptr %1107, i64 72
  %1233 = getelementptr inbounds nuw i8, ptr %1107, i64 76
  %1234 = getelementptr inbounds nuw i8, ptr %1107, i64 80
  %1235 = getelementptr inbounds nuw i8, ptr %1107, i64 84
  %1236 = getelementptr inbounds nuw i8, ptr %1107, i64 88
  %1237 = getelementptr inbounds nuw i8, ptr %1107, i64 92
  %1238 = getelementptr inbounds nuw i8, ptr %1107, i64 96
  %1239 = getelementptr inbounds nuw i8, ptr %1107, i64 100
  %1240 = getelementptr inbounds nuw i8, ptr %1107, i64 104
  %1241 = getelementptr inbounds nuw i8, ptr %1107, i64 108
  %1242 = getelementptr inbounds nuw i8, ptr %1107, i64 112
  %1243 = getelementptr inbounds nuw i8, ptr %1107, i64 116
  %1244 = getelementptr inbounds nuw i8, ptr %1107, i64 120
  %1245 = getelementptr inbounds nuw i8, ptr %1107, i64 124
  %1246 = getelementptr inbounds nuw i8, ptr %1107, i64 128
  %1247 = getelementptr inbounds nuw i8, ptr %1107, i64 132
  %1248 = getelementptr inbounds nuw i8, ptr %1107, i64 136
  %1249 = getelementptr inbounds nuw i8, ptr %1107, i64 140
  %1250 = getelementptr inbounds nuw i8, ptr %1107, i64 144
  %1251 = getelementptr inbounds nuw i8, ptr %1107, i64 148
  %1252 = getelementptr inbounds nuw i8, ptr %1107, i64 152
  %1253 = getelementptr inbounds nuw i8, ptr %1107, i64 156
  %1254 = getelementptr inbounds nuw i8, ptr %1107, i64 160
  %1255 = getelementptr inbounds nuw i8, ptr %1107, i64 164
  %1256 = getelementptr inbounds nuw i8, ptr %1107, i64 168
  %1257 = getelementptr inbounds nuw i8, ptr %1107, i64 172
  %1258 = getelementptr inbounds nuw i8, ptr %1107, i64 176
  %1259 = getelementptr inbounds nuw i8, ptr %1107, i64 180
  %1260 = getelementptr inbounds nuw i8, ptr %1107, i64 184
  %1261 = getelementptr inbounds nuw i8, ptr %1107, i64 188
  %1262 = getelementptr inbounds nuw i8, ptr %1107, i64 192
  %1263 = getelementptr inbounds nuw i8, ptr %1107, i64 196
  %1264 = getelementptr inbounds nuw i8, ptr %1107, i64 200
  %1265 = getelementptr inbounds nuw i8, ptr %1107, i64 204
  %1266 = getelementptr inbounds nuw i8, ptr %1107, i64 208
  %1267 = getelementptr inbounds nuw i8, ptr %1107, i64 212
  %1268 = getelementptr inbounds nuw i8, ptr %1107, i64 216
  %1269 = getelementptr inbounds nuw i8, ptr %1107, i64 220
  %1270 = getelementptr inbounds nuw i8, ptr %1107, i64 224
  %1271 = getelementptr inbounds nuw i8, ptr %1107, i64 228
  %1272 = getelementptr inbounds nuw i8, ptr %1107, i64 232
  %1273 = getelementptr inbounds nuw i8, ptr %1107, i64 236
  %1274 = getelementptr inbounds nuw i8, ptr %1107, i64 240
  %1275 = getelementptr inbounds nuw i8, ptr %1107, i64 244
  %1276 = getelementptr inbounds nuw i8, ptr %1107, i64 248
  %1277 = getelementptr inbounds nuw i8, ptr %1107, i64 252
  %1278 = getelementptr inbounds nuw i8, ptr %1107, i64 256
  %1279 = getelementptr inbounds nuw i8, ptr %1107, i64 260
  %1280 = getelementptr inbounds nuw i8, ptr %1107, i64 264
  %1281 = getelementptr inbounds nuw i8, ptr %1107, i64 268
  %1282 = getelementptr inbounds nuw i8, ptr %1107, i64 272
  %1283 = getelementptr inbounds nuw i8, ptr %1107, i64 276
  %1284 = getelementptr inbounds nuw i8, ptr %1107, i64 280
  %1285 = getelementptr inbounds nuw i8, ptr %1107, i64 284
  %1286 = getelementptr inbounds nuw i8, ptr %1107, i64 288
  %1287 = getelementptr inbounds nuw i8, ptr %1107, i64 292
  %1288 = getelementptr inbounds nuw i8, ptr %1107, i64 296
  %1289 = getelementptr inbounds nuw i8, ptr %1107, i64 300
  %1290 = getelementptr inbounds nuw i8, ptr %1107, i64 304
  %1291 = getelementptr inbounds nuw i8, ptr %1107, i64 308
  %1292 = getelementptr inbounds nuw i8, ptr %1107, i64 312
  %1293 = getelementptr inbounds nuw i8, ptr %1107, i64 316
  %1294 = getelementptr inbounds nuw i8, ptr %1107, i64 320
  %1295 = getelementptr inbounds nuw i8, ptr %1107, i64 324
  %1296 = getelementptr inbounds nuw i8, ptr %1107, i64 328
  %1297 = getelementptr inbounds nuw i8, ptr %1107, i64 332
  %1298 = getelementptr inbounds nuw i8, ptr %1107, i64 336
  %1299 = getelementptr inbounds nuw i8, ptr %1107, i64 340
  %1300 = getelementptr inbounds nuw i8, ptr %1107, i64 344
  %1301 = getelementptr inbounds nuw i8, ptr %1107, i64 348
  %1302 = getelementptr inbounds nuw i8, ptr %1107, i64 352
  %1303 = getelementptr inbounds nuw i8, ptr %1107, i64 356
  %1304 = getelementptr inbounds nuw i8, ptr %1107, i64 360
  %1305 = getelementptr inbounds nuw i8, ptr %1107, i64 364
  %1306 = getelementptr inbounds nuw i8, ptr %1107, i64 368
  %1307 = getelementptr inbounds nuw i8, ptr %1107, i64 372
  %1308 = getelementptr inbounds nuw i8, ptr %1107, i64 376
  %1309 = getelementptr inbounds nuw i8, ptr %1107, i64 380
  %1310 = getelementptr inbounds nuw i8, ptr %24, i64 4
  %1311 = getelementptr inbounds nuw i8, ptr %24, i64 8
  %1312 = getelementptr inbounds nuw i8, ptr %24, i64 12
  %1313 = getelementptr inbounds nuw i8, ptr %24, i64 16
  %1314 = getelementptr inbounds nuw i8, ptr %24, i64 20
  %1315 = getelementptr inbounds nuw i8, ptr %24, i64 24
  %1316 = getelementptr inbounds nuw i8, ptr %24, i64 28
  br label %1317

1317:                                             ; preds = %1058, %3721
  %1318 = phi double [ 0.000000e+00, %1058 ], [ %3731, %3721 ]
  %1319 = phi float [ %1073, %1058 ], [ %3729, %3721 ]
  %1320 = phi float [ %1072, %1058 ], [ %3725, %3721 ]
  %.pn241419 = phi ptr [ %1108, %1058 ], [ %3732, %3721 ]
  %1321 = phi i64 [ 0, %1058 ], [ %3733, %3721 ]
  store float %1320, ptr %1078, align 64, !tbaa !5
  store float %1319, ptr %1083, align 64, !tbaa !5
  %1322 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 448)
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 64 dereferenceable(384) %1088, i8 0, i64 384, i1 false), !tbaa !5
  %1323 = ptrtoint ptr %1322 to i64
  %1324 = add i64 %1323, 63
  %1325 = and i64 %1324, -64
  %1326 = inttoptr i64 %1325 to ptr
  %1327 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 384)
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(384) %1327, ptr noundef nonnull align 64 dereferenceable(384) %1088, i64 384, i1 false)
  %1328 = shl nuw nsw i64 %1321, 5
  %1329 = add nuw nsw i64 %1328, %.zext
  %1330 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 296
  %1331 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 292
  %1332 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 288
  %1333 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 200
  %1334 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 196
  %1335 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 192
  %1336 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 176
  %1337 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 172
  %1338 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 168
  %1339 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 92
  %1340 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 88
  %1341 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 84
  %1342 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 80
  %1343 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 76
  %1344 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 72
  %1345 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 68
  %1346 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 64
  %1347 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 60
  %1348 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 56
  %1349 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 52
  %1350 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 48
  %1351 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 44
  %1352 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 40
  %1353 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 36
  %1354 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 32
  %1355 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 28
  %1356 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 24
  %1357 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 8
  %1358 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 4
  %1359 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 20
  %1360 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 16
  %1361 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 12
  %1362 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 152
  %1363 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 148
  %1364 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 144
  %1365 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 104
  %1366 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 100
  %1367 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 96
  %1368 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 128
  %1369 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 124
  %1370 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 120
  %1371 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 236
  %1372 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 232
  %1373 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 228
  %1374 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 164
  %1375 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 160
  %1376 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 156
  %1377 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 116
  %1378 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 112
  %1379 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 108
  %1380 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 140
  %1381 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 136
  %1382 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 132
  %1383 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 224
  %1384 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 220
  %1385 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 216
  %1386 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 260
  %1387 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 256
  %1388 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 252
  %1389 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 188
  %1390 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 184
  %1391 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 180
  %1392 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 344
  %1393 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 340
  %1394 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 336
  %1395 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 284
  %1396 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 280
  %1397 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 276
  %1398 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 212
  %1399 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 208
  %1400 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 204
  %1401 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 248
  %1402 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 244
  %1403 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 240
  %1404 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 308
  %1405 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 304
  %1406 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 300
  %1407 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 272
  %1408 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 268
  %1409 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 264
  %1410 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 356
  %1411 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 352
  %1412 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 348
  %1413 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 320
  %1414 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 316
  %1415 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 312
  %1416 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 368
  %1417 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 364
  %1418 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 360
  %1419 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 332
  %1420 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 328
  %1421 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 324
  %1422 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 380
  %1423 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 376
  %1424 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 372
  %1425 = fpext float %1319 to double
  %1426 = fpext float %1320 to double
  %1427 = getelementptr inbounds nuw i8, ptr %1326, i64 4
  %1428 = getelementptr inbounds nuw i8, ptr %1326, i64 8
  %1429 = getelementptr inbounds nuw i8, ptr %1326, i64 12
  %1430 = getelementptr inbounds nuw i8, ptr %1326, i64 16
  %1431 = getelementptr inbounds nuw i8, ptr %1326, i64 20
  %1432 = getelementptr inbounds nuw i8, ptr %1326, i64 24
  %1433 = getelementptr inbounds nuw i8, ptr %1326, i64 28
  %1434 = getelementptr inbounds nuw i8, ptr %1326, i64 32
  %1435 = getelementptr inbounds nuw i8, ptr %1326, i64 36
  %1436 = getelementptr inbounds nuw i8, ptr %1326, i64 40
  %1437 = getelementptr inbounds nuw i8, ptr %1326, i64 44
  %1438 = getelementptr inbounds nuw i8, ptr %1326, i64 48
  %1439 = getelementptr inbounds nuw i8, ptr %1326, i64 52
  %1440 = getelementptr inbounds nuw i8, ptr %1326, i64 56
  %1441 = getelementptr inbounds nuw i8, ptr %1326, i64 60
  %1442 = getelementptr inbounds nuw i8, ptr %1326, i64 64
  %1443 = getelementptr inbounds nuw i8, ptr %1326, i64 68
  %1444 = getelementptr inbounds nuw i8, ptr %1326, i64 72
  %1445 = getelementptr inbounds nuw i8, ptr %1326, i64 76
  %1446 = getelementptr inbounds nuw i8, ptr %1326, i64 80
  %1447 = getelementptr inbounds nuw i8, ptr %1326, i64 84
  %1448 = getelementptr inbounds nuw i8, ptr %1326, i64 88
  %1449 = getelementptr inbounds nuw i8, ptr %1326, i64 92
  %1450 = getelementptr inbounds nuw i8, ptr %1326, i64 96
  %1451 = getelementptr inbounds nuw i8, ptr %1326, i64 100
  %1452 = getelementptr inbounds nuw i8, ptr %1326, i64 104
  %1453 = getelementptr inbounds nuw i8, ptr %1326, i64 108
  %1454 = getelementptr inbounds nuw i8, ptr %1326, i64 112
  %1455 = getelementptr inbounds nuw i8, ptr %1326, i64 116
  %1456 = getelementptr inbounds nuw i8, ptr %1326, i64 120
  %1457 = getelementptr inbounds nuw i8, ptr %1326, i64 124
  %1458 = getelementptr inbounds nuw i8, ptr %1326, i64 128
  %1459 = getelementptr inbounds nuw i8, ptr %1326, i64 132
  %1460 = getelementptr inbounds nuw i8, ptr %1326, i64 136
  %1461 = getelementptr inbounds nuw i8, ptr %1326, i64 140
  %1462 = getelementptr inbounds nuw i8, ptr %1326, i64 144
  %1463 = getelementptr inbounds nuw i8, ptr %1326, i64 148
  %1464 = getelementptr inbounds nuw i8, ptr %1326, i64 152
  %1465 = getelementptr inbounds nuw i8, ptr %1326, i64 156
  %1466 = getelementptr inbounds nuw i8, ptr %1326, i64 160
  %1467 = getelementptr inbounds nuw i8, ptr %1326, i64 164
  %1468 = getelementptr inbounds nuw i8, ptr %1326, i64 168
  %1469 = getelementptr inbounds nuw i8, ptr %1326, i64 172
  %1470 = getelementptr inbounds nuw i8, ptr %1326, i64 176
  %1471 = getelementptr inbounds nuw i8, ptr %1326, i64 180
  %1472 = getelementptr inbounds nuw i8, ptr %1326, i64 184
  %1473 = getelementptr inbounds nuw i8, ptr %1326, i64 188
  %1474 = getelementptr inbounds nuw i8, ptr %1326, i64 192
  %1475 = getelementptr inbounds nuw i8, ptr %1326, i64 196
  %1476 = getelementptr inbounds nuw i8, ptr %1326, i64 200
  %1477 = getelementptr inbounds nuw i8, ptr %1326, i64 204
  %1478 = getelementptr inbounds nuw i8, ptr %1326, i64 208
  %1479 = getelementptr inbounds nuw i8, ptr %1326, i64 212
  %1480 = getelementptr inbounds nuw i8, ptr %1326, i64 216
  %1481 = getelementptr inbounds nuw i8, ptr %1326, i64 220
  %1482 = getelementptr inbounds nuw i8, ptr %1326, i64 224
  %1483 = getelementptr inbounds nuw i8, ptr %1326, i64 228
  %1484 = getelementptr inbounds nuw i8, ptr %1326, i64 232
  %1485 = getelementptr inbounds nuw i8, ptr %1326, i64 236
  %1486 = getelementptr inbounds nuw i8, ptr %1326, i64 240
  %1487 = getelementptr inbounds nuw i8, ptr %1326, i64 244
  %1488 = getelementptr inbounds nuw i8, ptr %1326, i64 248
  %1489 = getelementptr inbounds nuw i8, ptr %1326, i64 252
  %1490 = getelementptr inbounds nuw i8, ptr %1326, i64 256
  %1491 = getelementptr inbounds nuw i8, ptr %1326, i64 260
  %1492 = getelementptr inbounds nuw i8, ptr %1326, i64 264
  %1493 = getelementptr inbounds nuw i8, ptr %1326, i64 268
  %1494 = getelementptr inbounds nuw i8, ptr %1326, i64 272
  %1495 = getelementptr inbounds nuw i8, ptr %1326, i64 276
  %1496 = getelementptr inbounds nuw i8, ptr %1326, i64 280
  %1497 = getelementptr inbounds nuw i8, ptr %1326, i64 284
  %1498 = getelementptr inbounds nuw i8, ptr %1326, i64 288
  %1499 = getelementptr inbounds nuw i8, ptr %1326, i64 292
  %1500 = getelementptr inbounds nuw i8, ptr %1326, i64 296
  %1501 = getelementptr inbounds nuw i8, ptr %1326, i64 300
  %1502 = getelementptr inbounds nuw i8, ptr %1326, i64 304
  %1503 = getelementptr inbounds nuw i8, ptr %1326, i64 308
  %1504 = getelementptr inbounds nuw i8, ptr %1326, i64 312
  %1505 = getelementptr inbounds nuw i8, ptr %1326, i64 316
  %1506 = getelementptr inbounds nuw i8, ptr %1326, i64 320
  %1507 = getelementptr inbounds nuw i8, ptr %1326, i64 324
  %1508 = getelementptr inbounds nuw i8, ptr %1326, i64 328
  %1509 = getelementptr inbounds nuw i8, ptr %1326, i64 332
  %1510 = getelementptr inbounds nuw i8, ptr %1326, i64 336
  %1511 = getelementptr inbounds nuw i8, ptr %1326, i64 340
  %1512 = getelementptr inbounds nuw i8, ptr %1326, i64 344
  %1513 = getelementptr inbounds nuw i8, ptr %1326, i64 348
  %1514 = getelementptr inbounds nuw i8, ptr %1326, i64 352
  %1515 = getelementptr inbounds nuw i8, ptr %1326, i64 356
  %1516 = getelementptr inbounds nuw i8, ptr %1326, i64 360
  %1517 = getelementptr inbounds nuw i8, ptr %1326, i64 364
  %1518 = getelementptr inbounds nuw i8, ptr %1326, i64 368
  %1519 = getelementptr inbounds nuw i8, ptr %1326, i64 372
  %1520 = getelementptr inbounds nuw i8, ptr %1326, i64 376
  %1521 = getelementptr inbounds nuw i8, ptr %1326, i64 380
  br label %.preheader332

.preheader332:                                    ; preds = %1317, %.preheader332
  %1522 = phi double [ 0.000000e+00, %1317 ], [ %2986, %.preheader332 ]
  %1523 = phi double [ 0.000000e+00, %1317 ], [ %2985, %.preheader332 ]
  %1524 = phi double [ 0.000000e+00, %1317 ], [ %2983, %.preheader332 ]
  %.pn297418 = phi ptr [ %1327, %1317 ], [ %2992, %.preheader332 ]
  %1525 = phi i64 [ 0, %1317 ], [ %2993, %.preheader332 ]
  %1526 = add nuw nsw i64 %1329, %1525
  %1527 = urem i64 %1526, 96
  %1528 = shl nuw nsw i64 %1527, 3
  %1529 = getelementptr inbounds nuw float, ptr %31, i64 %1527
  %1530 = load float, ptr %1529, align 4, !tbaa !5
  %1531 = getelementptr inbounds nuw float, ptr %36, i64 %1527
  %1532 = load float, ptr %1531, align 4, !tbaa !5
  %1533 = getelementptr inbounds nuw float, ptr %24, i64 %1528
  %1534 = load float, ptr %1533, align 4, !tbaa !5
  store float %1534, ptr %1093, align 64, !tbaa !5
  %1535 = getelementptr inbounds nuw i8, ptr %1533, i64 4
  %1536 = load float, ptr %1535, align 4, !tbaa !5
  store float %1536, ptr %1113, align 4, !tbaa !5
  %1537 = getelementptr inbounds nuw i8, ptr %1533, i64 8
  %1538 = load float, ptr %1537, align 4, !tbaa !5
  store float %1538, ptr %1114, align 8, !tbaa !5
  %1539 = getelementptr inbounds nuw i8, ptr %1533, i64 12
  %1540 = load float, ptr %1539, align 4, !tbaa !5
  store float %1540, ptr %1115, align 4, !tbaa !5
  %1541 = getelementptr inbounds nuw i8, ptr %1533, i64 16
  %1542 = load float, ptr %1541, align 4, !tbaa !5
  store float %1542, ptr %1116, align 16, !tbaa !5
  %1543 = getelementptr inbounds nuw i8, ptr %1533, i64 20
  %1544 = load float, ptr %1543, align 4, !tbaa !5
  store float %1544, ptr %1117, align 4, !tbaa !5
  %1545 = getelementptr inbounds nuw i8, ptr %1533, i64 24
  %1546 = load float, ptr %1545, align 4, !tbaa !5
  store float %1546, ptr %1118, align 8, !tbaa !5
  %1547 = getelementptr inbounds nuw i8, ptr %1533, i64 28
  %1548 = load float, ptr %1547, align 4, !tbaa !5
  store float %1548, ptr %1119, align 4, !tbaa !5
  %1549 = load float, ptr %1330, align 4, !tbaa !5
  %1550 = load float, ptr %1331, align 4, !tbaa !5
  %1551 = load float, ptr %1332, align 4, !tbaa !5
  %1552 = load float, ptr %1333, align 4, !tbaa !5
  %1553 = load float, ptr %1334, align 4, !tbaa !5
  %1554 = load float, ptr %1335, align 4, !tbaa !5
  %1555 = load float, ptr %1336, align 4, !tbaa !5
  %1556 = load float, ptr %1337, align 4, !tbaa !5
  %1557 = load float, ptr %1338, align 4, !tbaa !5
  %1558 = load float, ptr %1339, align 4, !tbaa !5
  %1559 = load float, ptr %1340, align 4, !tbaa !5
  %1560 = load float, ptr %1341, align 4, !tbaa !5
  %1561 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 96)
  %1562 = ptrtoint ptr %1561 to i64
  %1563 = add i64 %1562, 63
  %1564 = and i64 %1563, -64
  %1565 = inttoptr i64 %1564 to ptr
  store float 0x400921FB60000000, ptr %1565, align 64, !tbaa !5
  %1566 = getelementptr inbounds nuw i8, ptr %1565, i64 4
  store float 0x400921FB60000000, ptr %1566, align 4, !tbaa !5
  %1567 = getelementptr inbounds nuw i8, ptr %1565, i64 8
  store float 0x400921FB60000000, ptr %1567, align 8, !tbaa !5
  %1568 = getelementptr inbounds nuw i8, ptr %1565, i64 12
  store float 0x400921FB60000000, ptr %1568, align 4, !tbaa !5
  %1569 = getelementptr inbounds nuw i8, ptr %1565, i64 16
  store float 0x400921FB60000000, ptr %1569, align 16, !tbaa !5
  %1570 = getelementptr inbounds nuw i8, ptr %1565, i64 20
  store float 0x400921FB60000000, ptr %1570, align 4, !tbaa !5
  %1571 = getelementptr inbounds nuw i8, ptr %1565, i64 24
  store float 0x400921FB60000000, ptr %1571, align 8, !tbaa !5
  %1572 = getelementptr inbounds nuw i8, ptr %1565, i64 28
  store float 0x400921FB60000000, ptr %1572, align 4, !tbaa !5
  %1573 = load float, ptr %1093, align 64, !tbaa !5
  %1574 = fmul float %1573, 0x400921FB60000000
  store float %1574, ptr %1565, align 64, !tbaa !5
  %1575 = load float, ptr %1113, align 4, !tbaa !5
  %1576 = fmul float %1575, 0x400921FB60000000
  store float %1576, ptr %1566, align 4, !tbaa !5
  %1577 = load float, ptr %1114, align 8, !tbaa !5
  %1578 = fmul float %1577, 0x400921FB60000000
  store float %1578, ptr %1567, align 8, !tbaa !5
  %1579 = load float, ptr %1115, align 4, !tbaa !5
  %1580 = fmul float %1579, 0x400921FB60000000
  store float %1580, ptr %1568, align 4, !tbaa !5
  %1581 = load float, ptr %1116, align 16, !tbaa !5
  %1582 = fmul float %1581, 0x400921FB60000000
  store float %1582, ptr %1569, align 16, !tbaa !5
  %1583 = load float, ptr %1117, align 4, !tbaa !5
  %1584 = fmul float %1583, 0x400921FB60000000
  store float %1584, ptr %1570, align 4, !tbaa !5
  %1585 = load float, ptr %1118, align 8, !tbaa !5
  %1586 = fmul float %1585, 0x400921FB60000000
  store float %1586, ptr %1571, align 8, !tbaa !5
  %1587 = load float, ptr %1119, align 4, !tbaa !5
  %1588 = fmul float %1587, 0x400921FB60000000
  store float %1588, ptr %1572, align 4, !tbaa !5
  tail call void @__catalyst__rt__device_init(ptr nonnull @"/home/pichau/QAgents/.venv/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.so", ptr nonnull @LightningSimulator, ptr nonnull @"{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}", i64 0, i1 false)
  %1589 = tail call ptr @__catalyst__rt__qubit_allocate_array(i64 8)
  %1590 = tail call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %1589, i64 7)
  %1591 = load ptr, ptr %1590, align 8
  %1592 = fpext float %1588 to double
  tail call void @__catalyst__qis__RY(double %1592, ptr %1591, ptr null)
  %1593 = fpext float %1560 to double
  tail call void @__catalyst__qis__RZ(double %1593, ptr %1591, ptr null)
  %1594 = fpext float %1559 to double
  tail call void @__catalyst__qis__RY(double %1594, ptr %1591, ptr null)
  %1595 = fpext float %1558 to double
  tail call void @__catalyst__qis__RZ(double %1595, ptr %1591, ptr null)
  %1596 = load float, ptr %1342, align 4, !tbaa !5
  %1597 = load float, ptr %1343, align 4, !tbaa !5
  %1598 = load float, ptr %1344, align 4, !tbaa !5
  %1599 = load float, ptr %1571, align 8, !tbaa !5
  %1600 = tail call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %1589, i64 6)
  %1601 = load ptr, ptr %1600, align 8
  %1602 = fpext float %1599 to double
  tail call void @__catalyst__qis__RY(double %1602, ptr %1601, ptr null)
  %1603 = fpext float %1598 to double
  tail call void @__catalyst__qis__RZ(double %1603, ptr %1601, ptr null)
  %1604 = fpext float %1597 to double
  tail call void @__catalyst__qis__RY(double %1604, ptr %1601, ptr null)
  %1605 = fpext float %1596 to double
  tail call void @__catalyst__qis__RZ(double %1605, ptr %1601, ptr null)
  %1606 = load float, ptr %1345, align 4, !tbaa !5
  %1607 = load float, ptr %1346, align 4, !tbaa !5
  %1608 = load float, ptr %1347, align 4, !tbaa !5
  %1609 = load float, ptr %1570, align 4, !tbaa !5
  %1610 = tail call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %1589, i64 5)
  %1611 = load ptr, ptr %1610, align 8
  %1612 = fpext float %1609 to double
  tail call void @__catalyst__qis__RY(double %1612, ptr %1611, ptr null)
  %1613 = fpext float %1608 to double
  tail call void @__catalyst__qis__RZ(double %1613, ptr %1611, ptr null)
  %1614 = fpext float %1607 to double
  tail call void @__catalyst__qis__RY(double %1614, ptr %1611, ptr null)
  %1615 = fpext float %1606 to double
  tail call void @__catalyst__qis__RZ(double %1615, ptr %1611, ptr null)
  %1616 = load float, ptr %1348, align 4, !tbaa !5
  %1617 = load float, ptr %1349, align 4, !tbaa !5
  %1618 = load float, ptr %1350, align 4, !tbaa !5
  %1619 = load float, ptr %1569, align 16, !tbaa !5
  %1620 = tail call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %1589, i64 4)
  %1621 = load ptr, ptr %1620, align 8
  %1622 = fpext float %1619 to double
  tail call void @__catalyst__qis__RY(double %1622, ptr %1621, ptr null)
  %1623 = fpext float %1618 to double
  tail call void @__catalyst__qis__RZ(double %1623, ptr %1621, ptr null)
  %1624 = fpext float %1617 to double
  tail call void @__catalyst__qis__RY(double %1624, ptr %1621, ptr null)
  %1625 = fpext float %1616 to double
  tail call void @__catalyst__qis__RZ(double %1625, ptr %1621, ptr null)
  %1626 = load float, ptr %1351, align 4, !tbaa !5
  %1627 = load float, ptr %1352, align 4, !tbaa !5
  %1628 = load float, ptr %1353, align 4, !tbaa !5
  %1629 = load float, ptr %1568, align 4, !tbaa !5
  %1630 = tail call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %1589, i64 3)
  %1631 = load ptr, ptr %1630, align 8
  %1632 = fpext float %1629 to double
  tail call void @__catalyst__qis__RY(double %1632, ptr %1631, ptr null)
  %1633 = fpext float %1628 to double
  tail call void @__catalyst__qis__RZ(double %1633, ptr %1631, ptr null)
  %1634 = fpext float %1627 to double
  tail call void @__catalyst__qis__RY(double %1634, ptr %1631, ptr null)
  %1635 = fpext float %1626 to double
  tail call void @__catalyst__qis__RZ(double %1635, ptr %1631, ptr null)
  %1636 = load float, ptr %1354, align 4, !tbaa !5
  %1637 = load float, ptr %1355, align 4, !tbaa !5
  %1638 = load float, ptr %1356, align 4, !tbaa !5
  %1639 = load float, ptr %1567, align 8, !tbaa !5
  %1640 = tail call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %1589, i64 2)
  %1641 = load ptr, ptr %1640, align 8
  %1642 = fpext float %1639 to double
  tail call void @__catalyst__qis__RY(double %1642, ptr %1641, ptr null)
  %1643 = fpext float %1638 to double
  tail call void @__catalyst__qis__RZ(double %1643, ptr %1641, ptr null)
  %1644 = fpext float %1637 to double
  tail call void @__catalyst__qis__RY(double %1644, ptr %1641, ptr null)
  %1645 = fpext float %1636 to double
  tail call void @__catalyst__qis__RZ(double %1645, ptr %1641, ptr null)
  %1646 = load float, ptr %1357, align 4, !tbaa !5
  %1647 = load float, ptr %1358, align 4, !tbaa !5
  %1648 = load float, ptr %.pn241419, align 4, !tbaa !5
  %1649 = load float, ptr %1565, align 64, !tbaa !5
  %1650 = tail call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %1589, i64 0)
  %1651 = load ptr, ptr %1650, align 8
  %1652 = fpext float %1649 to double
  tail call void @__catalyst__qis__RY(double %1652, ptr %1651, ptr null)
  %1653 = fpext float %1648 to double
  tail call void @__catalyst__qis__RZ(double %1653, ptr %1651, ptr null)
  %1654 = fpext float %1647 to double
  tail call void @__catalyst__qis__RY(double %1654, ptr %1651, ptr null)
  %1655 = fpext float %1646 to double
  tail call void @__catalyst__qis__RZ(double %1655, ptr %1651, ptr null)
  %1656 = load float, ptr %1359, align 4, !tbaa !5
  %1657 = load float, ptr %1360, align 4, !tbaa !5
  %1658 = load float, ptr %1361, align 4, !tbaa !5
  %1659 = load float, ptr %1566, align 4, !tbaa !5
  tail call void @_mlir_memref_to_llvm_free(ptr %1561)
  %1660 = tail call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %1589, i64 1)
  %1661 = load ptr, ptr %1660, align 8
  %1662 = fpext float %1659 to double
  tail call void @__catalyst__qis__RY(double %1662, ptr %1661, ptr null)
  %1663 = fpext float %1658 to double
  tail call void @__catalyst__qis__RZ(double %1663, ptr %1661, ptr null)
  %1664 = fpext float %1657 to double
  tail call void @__catalyst__qis__RY(double %1664, ptr %1661, ptr null)
  %1665 = fpext float %1656 to double
  tail call void @__catalyst__qis__RZ(double %1665, ptr %1661, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %1651, ptr %1661, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %1661, ptr %1641, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %1641, ptr %1631, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %1631, ptr %1621, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %1621, ptr %1611, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %1611, ptr %1601, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %1601, ptr %1591, ptr null)
  %1666 = fpext float %1557 to double
  tail call void @__catalyst__qis__RZ(double %1666, ptr %1601, ptr null)
  %1667 = fpext float %1556 to double
  tail call void @__catalyst__qis__RY(double %1667, ptr %1601, ptr null)
  %1668 = fpext float %1555 to double
  tail call void @__catalyst__qis__RZ(double %1668, ptr %1601, ptr null)
  %1669 = load float, ptr %1362, align 4, !tbaa !5
  %1670 = load float, ptr %1363, align 4, !tbaa !5
  %1671 = load float, ptr %1364, align 4, !tbaa !5
  %1672 = fpext float %1671 to double
  tail call void @__catalyst__qis__RZ(double %1672, ptr %1621, ptr null)
  %1673 = fpext float %1670 to double
  tail call void @__catalyst__qis__RY(double %1673, ptr %1621, ptr null)
  %1674 = fpext float %1669 to double
  tail call void @__catalyst__qis__RZ(double %1674, ptr %1621, ptr null)
  %1675 = load float, ptr %1365, align 4, !tbaa !5
  %1676 = load float, ptr %1366, align 4, !tbaa !5
  %1677 = load float, ptr %1367, align 4, !tbaa !5
  tail call void @__catalyst__qis__CNOT(ptr %1591, ptr %1651, ptr null)
  %1678 = fpext float %1677 to double
  tail call void @__catalyst__qis__RZ(double %1678, ptr %1651, ptr null)
  %1679 = fpext float %1676 to double
  tail call void @__catalyst__qis__RY(double %1679, ptr %1651, ptr null)
  %1680 = fpext float %1675 to double
  tail call void @__catalyst__qis__RZ(double %1680, ptr %1651, ptr null)
  %1681 = load float, ptr %1368, align 4, !tbaa !5
  %1682 = load float, ptr %1369, align 4, !tbaa !5
  %1683 = load float, ptr %1370, align 4, !tbaa !5
  %1684 = fpext float %1683 to double
  tail call void @__catalyst__qis__RZ(double %1684, ptr %1641, ptr null)
  %1685 = fpext float %1682 to double
  tail call void @__catalyst__qis__RY(double %1685, ptr %1641, ptr null)
  %1686 = fpext float %1681 to double
  tail call void @__catalyst__qis__RZ(double %1686, ptr %1641, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %1651, ptr %1641, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %1641, ptr %1621, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %1621, ptr %1601, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %1601, ptr %1651, ptr null)
  %1687 = fpext float %1554 to double
  tail call void @__catalyst__qis__RZ(double %1687, ptr %1651, ptr null)
  %1688 = fpext float %1553 to double
  tail call void @__catalyst__qis__RY(double %1688, ptr %1651, ptr null)
  %1689 = fpext float %1552 to double
  tail call void @__catalyst__qis__RZ(double %1689, ptr %1651, ptr null)
  %1690 = load float, ptr %1371, align 4, !tbaa !5
  %1691 = load float, ptr %1372, align 4, !tbaa !5
  %1692 = load float, ptr %1373, align 4, !tbaa !5
  %1693 = load float, ptr %1374, align 4, !tbaa !5
  %1694 = load float, ptr %1375, align 4, !tbaa !5
  %1695 = load float, ptr %1376, align 4, !tbaa !5
  %1696 = fpext float %1695 to double
  tail call void @__catalyst__qis__RZ(double %1696, ptr %1611, ptr null)
  %1697 = fpext float %1694 to double
  tail call void @__catalyst__qis__RY(double %1697, ptr %1611, ptr null)
  %1698 = fpext float %1693 to double
  tail call void @__catalyst__qis__RZ(double %1698, ptr %1611, ptr null)
  %1699 = load float, ptr %1377, align 4, !tbaa !5
  %1700 = load float, ptr %1378, align 4, !tbaa !5
  %1701 = load float, ptr %1379, align 4, !tbaa !5
  %1702 = fpext float %1701 to double
  tail call void @__catalyst__qis__RZ(double %1702, ptr %1661, ptr null)
  %1703 = fpext float %1700 to double
  tail call void @__catalyst__qis__RY(double %1703, ptr %1661, ptr null)
  %1704 = fpext float %1699 to double
  tail call void @__catalyst__qis__RZ(double %1704, ptr %1661, ptr null)
  %1705 = load float, ptr %1380, align 4, !tbaa !5
  %1706 = load float, ptr %1381, align 4, !tbaa !5
  %1707 = load float, ptr %1382, align 4, !tbaa !5
  %1708 = fpext float %1707 to double
  tail call void @__catalyst__qis__RZ(double %1708, ptr %1631, ptr null)
  %1709 = fpext float %1706 to double
  tail call void @__catalyst__qis__RY(double %1709, ptr %1631, ptr null)
  %1710 = fpext float %1705 to double
  tail call void @__catalyst__qis__RZ(double %1710, ptr %1631, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %1661, ptr %1631, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %1631, ptr %1611, ptr null)
  %1711 = fpext float %1692 to double
  tail call void @__catalyst__qis__RZ(double %1711, ptr %1631, ptr null)
  %1712 = fpext float %1691 to double
  tail call void @__catalyst__qis__RY(double %1712, ptr %1631, ptr null)
  %1713 = fpext float %1690 to double
  tail call void @__catalyst__qis__RZ(double %1713, ptr %1631, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %1651, ptr %1631, ptr null)
  %1714 = load float, ptr %1383, align 4, !tbaa !5
  %1715 = load float, ptr %1384, align 4, !tbaa !5
  %1716 = load float, ptr %1385, align 4, !tbaa !5
  %1717 = fpext float %1716 to double
  tail call void @__catalyst__qis__RZ(double %1717, ptr %1641, ptr null)
  %1718 = fpext float %1715 to double
  tail call void @__catalyst__qis__RY(double %1718, ptr %1641, ptr null)
  %1719 = fpext float %1714 to double
  tail call void @__catalyst__qis__RZ(double %1719, ptr %1641, ptr null)
  %1720 = load float, ptr %1386, align 4, !tbaa !5
  %1721 = load float, ptr %1387, align 4, !tbaa !5
  %1722 = load float, ptr %1388, align 4, !tbaa !5
  %1723 = load float, ptr %1389, align 4, !tbaa !5
  %1724 = load float, ptr %1390, align 4, !tbaa !5
  %1725 = load float, ptr %1391, align 4, !tbaa !5
  %1726 = fpext float %1725 to double
  tail call void @__catalyst__qis__RZ(double %1726, ptr %1591, ptr null)
  %1727 = fpext float %1724 to double
  tail call void @__catalyst__qis__RY(double %1727, ptr %1591, ptr null)
  %1728 = fpext float %1723 to double
  tail call void @__catalyst__qis__RZ(double %1728, ptr %1591, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %1611, ptr %1591, ptr null)
  %1729 = fpext float %1722 to double
  tail call void @__catalyst__qis__RZ(double %1729, ptr %1611, ptr null)
  %1730 = fpext float %1721 to double
  tail call void @__catalyst__qis__RY(double %1730, ptr %1611, ptr null)
  %1731 = fpext float %1720 to double
  tail call void @__catalyst__qis__RZ(double %1731, ptr %1611, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %1641, ptr %1611, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %1611, ptr %1651, ptr null)
  %1732 = fpext float %1551 to double
  tail call void @__catalyst__qis__RZ(double %1732, ptr %1651, ptr null)
  %1733 = fpext float %1550 to double
  tail call void @__catalyst__qis__RY(double %1733, ptr %1651, ptr null)
  %1734 = fpext float %1549 to double
  tail call void @__catalyst__qis__RZ(double %1734, ptr %1651, ptr null)
  %1735 = load float, ptr %1392, align 4, !tbaa !5
  %1736 = load float, ptr %1393, align 4, !tbaa !5
  %1737 = load float, ptr %1394, align 4, !tbaa !5
  %1738 = load float, ptr %1395, align 4, !tbaa !5
  %1739 = load float, ptr %1396, align 4, !tbaa !5
  %1740 = load float, ptr %1397, align 4, !tbaa !5
  tail call void @__catalyst__qis__CNOT(ptr %1591, ptr %1661, ptr null)
  %1741 = fpext float %1740 to double
  tail call void @__catalyst__qis__RZ(double %1741, ptr %1591, ptr null)
  %1742 = fpext float %1739 to double
  tail call void @__catalyst__qis__RY(double %1742, ptr %1591, ptr null)
  %1743 = fpext float %1738 to double
  tail call void @__catalyst__qis__RZ(double %1743, ptr %1591, ptr null)
  %1744 = load float, ptr %1398, align 4, !tbaa !5
  %1745 = load float, ptr %1399, align 4, !tbaa !5
  %1746 = load float, ptr %1400, align 4, !tbaa !5
  %1747 = fpext float %1746 to double
  tail call void @__catalyst__qis__RZ(double %1747, ptr %1661, ptr null)
  %1748 = fpext float %1745 to double
  tail call void @__catalyst__qis__RY(double %1748, ptr %1661, ptr null)
  %1749 = fpext float %1744 to double
  tail call void @__catalyst__qis__RZ(double %1749, ptr %1661, ptr null)
  %1750 = load float, ptr %1401, align 4, !tbaa !5
  %1751 = load float, ptr %1402, align 4, !tbaa !5
  %1752 = load float, ptr %1403, align 4, !tbaa !5
  %1753 = fpext float %1752 to double
  tail call void @__catalyst__qis__RZ(double %1753, ptr %1621, ptr null)
  %1754 = fpext float %1751 to double
  tail call void @__catalyst__qis__RY(double %1754, ptr %1621, ptr null)
  %1755 = fpext float %1750 to double
  tail call void @__catalyst__qis__RZ(double %1755, ptr %1621, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %1661, ptr %1621, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %1621, ptr %1591, ptr null)
  %1756 = fpext float %1737 to double
  tail call void @__catalyst__qis__RZ(double %1756, ptr %1621, ptr null)
  %1757 = fpext float %1736 to double
  tail call void @__catalyst__qis__RY(double %1757, ptr %1621, ptr null)
  %1758 = fpext float %1735 to double
  tail call void @__catalyst__qis__RZ(double %1758, ptr %1621, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %1651, ptr %1621, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %1621, ptr %1651, ptr null)
  %1759 = load float, ptr %1404, align 4, !tbaa !5
  %1760 = load float, ptr %1405, align 4, !tbaa !5
  %1761 = load float, ptr %1406, align 4, !tbaa !5
  %1762 = load float, ptr %1407, align 4, !tbaa !5
  %1763 = load float, ptr %1408, align 4, !tbaa !5
  %1764 = load float, ptr %1409, align 4, !tbaa !5
  %1765 = fpext float %1764 to double
  tail call void @__catalyst__qis__RZ(double %1765, ptr %1601, ptr null)
  %1766 = fpext float %1763 to double
  tail call void @__catalyst__qis__RY(double %1766, ptr %1601, ptr null)
  %1767 = fpext float %1762 to double
  tail call void @__catalyst__qis__RZ(double %1767, ptr %1601, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %1631, ptr %1601, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %1601, ptr %1661, ptr null)
  %1768 = fpext float %1761 to double
  tail call void @__catalyst__qis__RZ(double %1768, ptr %1661, ptr null)
  %1769 = fpext float %1760 to double
  tail call void @__catalyst__qis__RY(double %1769, ptr %1661, ptr null)
  %1770 = fpext float %1759 to double
  tail call void @__catalyst__qis__RZ(double %1770, ptr %1661, ptr null)
  %1771 = load float, ptr %1410, align 4, !tbaa !5
  %1772 = load float, ptr %1411, align 4, !tbaa !5
  %1773 = load float, ptr %1412, align 4, !tbaa !5
  %1774 = fpext float %1773 to double
  tail call void @__catalyst__qis__RZ(double %1774, ptr %1611, ptr null)
  %1775 = fpext float %1772 to double
  tail call void @__catalyst__qis__RY(double %1775, ptr %1611, ptr null)
  %1776 = fpext float %1771 to double
  tail call void @__catalyst__qis__RZ(double %1776, ptr %1611, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %1661, ptr %1611, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %1611, ptr %1661, ptr null)
  %1777 = load float, ptr %1413, align 4, !tbaa !5
  %1778 = load float, ptr %1414, align 4, !tbaa !5
  %1779 = load float, ptr %1415, align 4, !tbaa !5
  tail call void @__catalyst__qis__CNOT(ptr %1591, ptr %1641, ptr null)
  %1780 = fpext float %1779 to double
  tail call void @__catalyst__qis__RZ(double %1780, ptr %1641, ptr null)
  %1781 = fpext float %1778 to double
  tail call void @__catalyst__qis__RY(double %1781, ptr %1641, ptr null)
  %1782 = fpext float %1777 to double
  tail call void @__catalyst__qis__RZ(double %1782, ptr %1641, ptr null)
  %1783 = load float, ptr %1416, align 4, !tbaa !5
  %1784 = load float, ptr %1417, align 4, !tbaa !5
  %1785 = load float, ptr %1418, align 4, !tbaa !5
  %1786 = fpext float %1785 to double
  tail call void @__catalyst__qis__RZ(double %1786, ptr %1601, ptr null)
  %1787 = fpext float %1784 to double
  tail call void @__catalyst__qis__RY(double %1787, ptr %1601, ptr null)
  %1788 = fpext float %1783 to double
  tail call void @__catalyst__qis__RZ(double %1788, ptr %1601, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %1641, ptr %1601, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %1601, ptr %1641, ptr null)
  %1789 = load float, ptr %1419, align 4, !tbaa !5
  %1790 = load float, ptr %1420, align 4, !tbaa !5
  %1791 = load float, ptr %1421, align 4, !tbaa !5
  %1792 = fpext float %1791 to double
  tail call void @__catalyst__qis__RZ(double %1792, ptr %1631, ptr null)
  %1793 = fpext float %1790 to double
  tail call void @__catalyst__qis__RY(double %1793, ptr %1631, ptr null)
  %1794 = fpext float %1789 to double
  tail call void @__catalyst__qis__RZ(double %1794, ptr %1631, ptr null)
  %1795 = load float, ptr %1422, align 4, !tbaa !5
  %1796 = load float, ptr %1423, align 4, !tbaa !5
  %1797 = load float, ptr %1424, align 4, !tbaa !5
  %1798 = fpext float %1797 to double
  tail call void @__catalyst__qis__RZ(double %1798, ptr %1591, ptr null)
  %1799 = fpext float %1796 to double
  tail call void @__catalyst__qis__RY(double %1799, ptr %1591, ptr null)
  %1800 = fpext float %1795 to double
  tail call void @__catalyst__qis__RZ(double %1800, ptr %1591, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %1631, ptr %1591, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %1591, ptr %1631, ptr null)
  %1801 = tail call i64 @__catalyst__qis__NamedObs(i64 3, ptr %1651)
  %1802 = tail call double @__catalyst__qis__Expval(i64 %1801)
  %1803 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %1804 = ptrtoint ptr %1803 to i64
  %1805 = add i64 %1804, 63
  %1806 = and i64 %1805, -64
  %1807 = inttoptr i64 %1806 to ptr
  store double %1802, ptr %1807, align 64, !tbaa !7
  tail call void @__catalyst__rt__qubit_release_array(ptr %1589)
  tail call void @__catalyst__rt__device_release()
  %1808 = load double, ptr %1807, align 64, !tbaa !7
  store double 1.000000e+00, ptr %1098, align 64, !tbaa !7
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 1 dereferenceable(384) %1099, i8 0, i64 384, i1 false)
  store i32 0, ptr %1100, align 1
  store i32 0, ptr %1101, align 1
  %1809 = load float, ptr %36, align 4, !tbaa !5, !alias.scope !9, !noalias !12
  %1810 = load float, ptr %31, align 4, !tbaa !5, !alias.scope !14, !noalias !17
  %1811 = load float, ptr %1078, align 64, !tbaa !5, !alias.scope !19, !noalias !22
  %1812 = load float, ptr %1083, align 64, !tbaa !5, !alias.scope !24, !noalias !27
  %calloc.i = tail call dereferenceable_or_null(40) ptr @calloc(i64 1, i64 40)
  %1813 = tail call noalias nonnull dereferenceable(40) dereferenceable_or_null(40) ptr @malloc(i64 40), !enzyme_fromstack !29
  %1814 = tail call noalias nonnull dereferenceable(40) dereferenceable_or_null(40) ptr @malloc(i64 40), !enzyme_fromstack !29
  %calloc4.i = tail call dereferenceable_or_null(24) ptr @calloc(i64 1, i64 24)
  %1815 = tail call noalias nonnull dereferenceable(24) dereferenceable_or_null(24) ptr @malloc(i64 24), !enzyme_fromstack !29
  %1816 = tail call noalias nonnull dereferenceable(72) dereferenceable_or_null(72) ptr @malloc(i64 72), !enzyme_fromstack !29
  %"'mi7.i" = tail call noalias nonnull ptr @_mlir_memref_to_llvm_alloc(i64 832) #12
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 1 dereferenceable(832) %"'mi7.i", i8 0, i64 832, i1 false)
  %1817 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 832) #12
  %1818 = load float, ptr %1330, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %1819 = load float, ptr %1331, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %1820 = load float, ptr %1332, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %1821 = load float, ptr %1333, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %1822 = load float, ptr %1334, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %1823 = load float, ptr %1335, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %1824 = load float, ptr %1336, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %1825 = load float, ptr %1337, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %1826 = load float, ptr %1338, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %1827 = load float, ptr %1339, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %1828 = load float, ptr %1340, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %1829 = load float, ptr %1341, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %1830 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 96) #12
  %1831 = ptrtoint ptr %1830 to i64
  %1832 = add i64 %1831, 63
  %1833 = and i64 %1832, -64
  %1834 = inttoptr i64 %1833 to ptr
  store float 0x400921FB60000000, ptr %1834, align 64, !tbaa !5
  %1835 = getelementptr inbounds nuw i8, ptr %1834, i64 4
  store float 0x400921FB60000000, ptr %1835, align 4, !tbaa !5
  %1836 = getelementptr inbounds nuw i8, ptr %1834, i64 8
  store float 0x400921FB60000000, ptr %1836, align 8, !tbaa !5
  %1837 = getelementptr inbounds nuw i8, ptr %1834, i64 12
  store float 0x400921FB60000000, ptr %1837, align 4, !tbaa !5
  %1838 = getelementptr inbounds nuw i8, ptr %1834, i64 16
  store float 0x400921FB60000000, ptr %1838, align 16, !tbaa !5
  %1839 = getelementptr inbounds nuw i8, ptr %1834, i64 20
  store float 0x400921FB60000000, ptr %1839, align 4, !tbaa !5
  %1840 = getelementptr inbounds nuw i8, ptr %1834, i64 24
  store float 0x400921FB60000000, ptr %1840, align 8, !tbaa !5
  %1841 = getelementptr inbounds nuw i8, ptr %1834, i64 28
  store float 0x400921FB60000000, ptr %1841, align 4, !tbaa !5
  %1842 = load float, ptr %24, align 4, !tbaa !5, !alias.scope !35, !noalias !38
  %1843 = fmul float %1842, 0x400921FB60000000
  store float %1843, ptr %1834, align 64, !tbaa !5
  %1844 = load float, ptr %1310, align 4, !tbaa !5, !alias.scope !35, !noalias !38
  %1845 = fmul float %1844, 0x400921FB60000000
  store float %1845, ptr %1835, align 4, !tbaa !5
  %1846 = load float, ptr %1311, align 4, !tbaa !5, !alias.scope !35, !noalias !38
  %1847 = fmul float %1846, 0x400921FB60000000
  store float %1847, ptr %1836, align 8, !tbaa !5
  %1848 = load float, ptr %1312, align 4, !tbaa !5, !alias.scope !35, !noalias !38
  %1849 = fmul float %1848, 0x400921FB60000000
  store float %1849, ptr %1837, align 4, !tbaa !5
  %1850 = load float, ptr %1313, align 4, !tbaa !5, !alias.scope !35, !noalias !38
  %1851 = fmul float %1850, 0x400921FB60000000
  store float %1851, ptr %1838, align 16, !tbaa !5
  %1852 = load float, ptr %1314, align 4, !tbaa !5, !alias.scope !35, !noalias !38
  %1853 = fmul float %1852, 0x400921FB60000000
  store float %1853, ptr %1839, align 4, !tbaa !5
  %1854 = load float, ptr %1315, align 4, !tbaa !5, !alias.scope !35, !noalias !38
  %1855 = fmul float %1854, 0x400921FB60000000
  store float %1855, ptr %1840, align 8, !tbaa !5
  %1856 = load float, ptr %1316, align 4, !tbaa !5, !alias.scope !35, !noalias !38
  %1857 = fmul float %1856, 0x400921FB60000000
  store float %1857, ptr %1841, align 4, !tbaa !5
  %1858 = fpext float %1857 to double
  store double %1858, ptr %1817, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %1859 = fpext float %1829 to double
  %1860 = getelementptr inbounds nuw i8, ptr %1817, i64 8
  store double %1859, ptr %1860, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %1861 = fpext float %1828 to double
  %1862 = getelementptr inbounds nuw i8, ptr %1817, i64 16
  store double %1861, ptr %1862, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %1863 = fpext float %1827 to double
  %1864 = getelementptr inbounds nuw i8, ptr %1817, i64 24
  store double %1863, ptr %1864, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %1865 = load float, ptr %1342, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %1866 = load float, ptr %1343, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %1867 = load float, ptr %1344, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %1868 = fpext float %1855 to double
  %1869 = getelementptr inbounds nuw i8, ptr %1817, i64 32
  store double %1868, ptr %1869, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %1870 = fpext float %1867 to double
  %1871 = getelementptr inbounds nuw i8, ptr %1817, i64 40
  store double %1870, ptr %1871, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %1872 = fpext float %1866 to double
  %1873 = getelementptr inbounds nuw i8, ptr %1817, i64 48
  store double %1872, ptr %1873, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %1874 = fpext float %1865 to double
  %1875 = getelementptr inbounds nuw i8, ptr %1817, i64 56
  store double %1874, ptr %1875, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %1876 = load float, ptr %1345, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %1877 = load float, ptr %1346, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %1878 = load float, ptr %1347, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %1879 = fpext float %1853 to double
  %1880 = getelementptr inbounds nuw i8, ptr %1817, i64 64
  store double %1879, ptr %1880, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %1881 = fpext float %1878 to double
  %1882 = getelementptr inbounds nuw i8, ptr %1817, i64 72
  store double %1881, ptr %1882, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %1883 = fpext float %1877 to double
  %1884 = getelementptr inbounds nuw i8, ptr %1817, i64 80
  store double %1883, ptr %1884, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %1885 = fpext float %1876 to double
  %1886 = getelementptr inbounds nuw i8, ptr %1817, i64 88
  store double %1885, ptr %1886, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %1887 = load float, ptr %1348, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %1888 = load float, ptr %1349, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %1889 = load float, ptr %1350, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %1890 = fpext float %1851 to double
  %1891 = getelementptr inbounds nuw i8, ptr %1817, i64 96
  store double %1890, ptr %1891, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %1892 = fpext float %1889 to double
  %1893 = getelementptr inbounds nuw i8, ptr %1817, i64 104
  store double %1892, ptr %1893, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %1894 = fpext float %1888 to double
  %1895 = getelementptr inbounds nuw i8, ptr %1817, i64 112
  store double %1894, ptr %1895, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %1896 = fpext float %1887 to double
  %1897 = getelementptr inbounds nuw i8, ptr %1817, i64 120
  store double %1896, ptr %1897, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %1898 = load float, ptr %1351, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %1899 = load float, ptr %1352, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %1900 = load float, ptr %1353, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %1901 = fpext float %1849 to double
  %1902 = getelementptr inbounds nuw i8, ptr %1817, i64 128
  store double %1901, ptr %1902, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %1903 = fpext float %1900 to double
  %1904 = getelementptr inbounds nuw i8, ptr %1817, i64 136
  store double %1903, ptr %1904, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %1905 = fpext float %1899 to double
  %1906 = getelementptr inbounds nuw i8, ptr %1817, i64 144
  store double %1905, ptr %1906, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %1907 = fpext float %1898 to double
  %1908 = getelementptr inbounds nuw i8, ptr %1817, i64 152
  store double %1907, ptr %1908, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %1909 = load float, ptr %1354, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %1910 = load float, ptr %1355, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %1911 = load float, ptr %1356, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %1912 = fpext float %1847 to double
  %1913 = getelementptr inbounds nuw i8, ptr %1817, i64 160
  store double %1912, ptr %1913, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %1914 = fpext float %1911 to double
  %1915 = getelementptr inbounds nuw i8, ptr %1817, i64 168
  store double %1914, ptr %1915, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %1916 = fpext float %1910 to double
  %1917 = getelementptr inbounds nuw i8, ptr %1817, i64 176
  store double %1916, ptr %1917, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %1918 = fpext float %1909 to double
  %1919 = getelementptr inbounds nuw i8, ptr %1817, i64 184
  store double %1918, ptr %1919, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %1920 = load float, ptr %1357, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %1921 = load float, ptr %1358, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %1922 = load float, ptr %.pn241419, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %1923 = fpext float %1843 to double
  %1924 = getelementptr inbounds nuw i8, ptr %1817, i64 192
  store double %1923, ptr %1924, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %1925 = fpext float %1922 to double
  %1926 = getelementptr inbounds nuw i8, ptr %1817, i64 200
  store double %1925, ptr %1926, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %1927 = fpext float %1921 to double
  %1928 = getelementptr inbounds nuw i8, ptr %1817, i64 208
  store double %1927, ptr %1928, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %1929 = fpext float %1920 to double
  %1930 = getelementptr inbounds nuw i8, ptr %1817, i64 216
  store double %1929, ptr %1930, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %1931 = load float, ptr %1359, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %1932 = load float, ptr %1360, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %1933 = load float, ptr %1361, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  tail call void @_mlir_memref_to_llvm_free(ptr %1830) #12
  %1934 = fpext float %1845 to double
  %1935 = getelementptr inbounds nuw i8, ptr %1817, i64 224
  store double %1934, ptr %1935, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %1936 = fpext float %1933 to double
  %1937 = getelementptr inbounds nuw i8, ptr %1817, i64 232
  store double %1936, ptr %1937, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %1938 = fpext float %1932 to double
  %1939 = getelementptr inbounds nuw i8, ptr %1817, i64 240
  store double %1938, ptr %1939, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %1940 = fpext float %1931 to double
  %1941 = getelementptr inbounds nuw i8, ptr %1817, i64 248
  store double %1940, ptr %1941, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %1942 = fpext float %1826 to double
  %1943 = getelementptr inbounds nuw i8, ptr %1817, i64 256
  store double %1942, ptr %1943, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %1944 = fpext float %1825 to double
  %1945 = getelementptr inbounds nuw i8, ptr %1817, i64 264
  store double %1944, ptr %1945, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %1946 = fpext float %1824 to double
  %1947 = getelementptr inbounds nuw i8, ptr %1817, i64 272
  store double %1946, ptr %1947, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %1948 = load float, ptr %1362, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %1949 = load float, ptr %1363, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %1950 = load float, ptr %1364, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %1951 = fpext float %1950 to double
  %1952 = getelementptr inbounds nuw i8, ptr %1817, i64 280
  store double %1951, ptr %1952, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %1953 = fpext float %1949 to double
  %1954 = getelementptr inbounds nuw i8, ptr %1817, i64 288
  store double %1953, ptr %1954, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %1955 = fpext float %1948 to double
  %1956 = getelementptr inbounds nuw i8, ptr %1817, i64 296
  store double %1955, ptr %1956, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %1957 = load float, ptr %1365, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %1958 = load float, ptr %1366, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %1959 = load float, ptr %1367, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %1960 = fpext float %1959 to double
  %1961 = getelementptr inbounds nuw i8, ptr %1817, i64 304
  store double %1960, ptr %1961, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %1962 = fpext float %1958 to double
  %1963 = getelementptr inbounds nuw i8, ptr %1817, i64 312
  store double %1962, ptr %1963, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %1964 = fpext float %1957 to double
  %1965 = getelementptr inbounds nuw i8, ptr %1817, i64 320
  store double %1964, ptr %1965, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %1966 = load float, ptr %1368, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %1967 = load float, ptr %1369, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %1968 = load float, ptr %1370, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %1969 = fpext float %1968 to double
  %1970 = getelementptr inbounds nuw i8, ptr %1817, i64 328
  store double %1969, ptr %1970, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %1971 = fpext float %1967 to double
  %1972 = getelementptr inbounds nuw i8, ptr %1817, i64 336
  store double %1971, ptr %1972, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %1973 = fpext float %1966 to double
  %1974 = getelementptr inbounds nuw i8, ptr %1817, i64 344
  store double %1973, ptr %1974, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %1975 = fpext float %1823 to double
  %1976 = getelementptr inbounds nuw i8, ptr %1817, i64 352
  store double %1975, ptr %1976, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %1977 = fpext float %1822 to double
  %1978 = getelementptr inbounds nuw i8, ptr %1817, i64 360
  store double %1977, ptr %1978, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %1979 = fpext float %1821 to double
  %1980 = getelementptr inbounds nuw i8, ptr %1817, i64 368
  store double %1979, ptr %1980, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %1981 = load float, ptr %1371, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %1982 = load float, ptr %1372, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %1983 = load float, ptr %1373, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %1984 = load float, ptr %1374, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %1985 = load float, ptr %1375, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %1986 = load float, ptr %1376, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %1987 = fpext float %1986 to double
  %1988 = getelementptr inbounds nuw i8, ptr %1817, i64 376
  store double %1987, ptr %1988, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %1989 = fpext float %1985 to double
  %1990 = getelementptr inbounds nuw i8, ptr %1817, i64 384
  store double %1989, ptr %1990, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %1991 = fpext float %1984 to double
  %1992 = getelementptr inbounds nuw i8, ptr %1817, i64 392
  store double %1991, ptr %1992, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %1993 = load float, ptr %1377, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %1994 = load float, ptr %1378, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %1995 = load float, ptr %1379, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %1996 = fpext float %1995 to double
  %1997 = getelementptr inbounds nuw i8, ptr %1817, i64 400
  store double %1996, ptr %1997, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %1998 = fpext float %1994 to double
  %1999 = getelementptr inbounds nuw i8, ptr %1817, i64 408
  store double %1998, ptr %1999, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %2000 = fpext float %1993 to double
  %2001 = getelementptr inbounds nuw i8, ptr %1817, i64 416
  store double %2000, ptr %2001, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %2002 = load float, ptr %1380, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %2003 = load float, ptr %1381, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %2004 = load float, ptr %1382, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %2005 = fpext float %2004 to double
  %2006 = getelementptr inbounds nuw i8, ptr %1817, i64 424
  store double %2005, ptr %2006, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %2007 = fpext float %2003 to double
  %2008 = getelementptr inbounds nuw i8, ptr %1817, i64 432
  store double %2007, ptr %2008, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %2009 = fpext float %2002 to double
  %2010 = getelementptr inbounds nuw i8, ptr %1817, i64 440
  store double %2009, ptr %2010, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %2011 = fpext float %1983 to double
  %2012 = getelementptr inbounds nuw i8, ptr %1817, i64 448
  store double %2011, ptr %2012, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %2013 = fpext float %1982 to double
  %2014 = getelementptr inbounds nuw i8, ptr %1817, i64 456
  store double %2013, ptr %2014, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %2015 = fpext float %1981 to double
  %2016 = getelementptr inbounds nuw i8, ptr %1817, i64 464
  store double %2015, ptr %2016, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %2017 = load float, ptr %1383, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %2018 = load float, ptr %1384, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %2019 = load float, ptr %1385, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %2020 = fpext float %2019 to double
  %2021 = getelementptr inbounds nuw i8, ptr %1817, i64 472
  store double %2020, ptr %2021, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %2022 = fpext float %2018 to double
  %2023 = getelementptr inbounds nuw i8, ptr %1817, i64 480
  store double %2022, ptr %2023, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %2024 = fpext float %2017 to double
  %2025 = getelementptr inbounds nuw i8, ptr %1817, i64 488
  store double %2024, ptr %2025, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %2026 = load float, ptr %1386, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %2027 = load float, ptr %1387, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %2028 = load float, ptr %1388, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %2029 = load float, ptr %1389, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %2030 = load float, ptr %1390, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %2031 = load float, ptr %1391, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %2032 = fpext float %2031 to double
  %2033 = getelementptr inbounds nuw i8, ptr %1817, i64 496
  store double %2032, ptr %2033, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %2034 = fpext float %2030 to double
  %2035 = getelementptr inbounds nuw i8, ptr %1817, i64 504
  store double %2034, ptr %2035, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %2036 = fpext float %2029 to double
  %2037 = getelementptr inbounds nuw i8, ptr %1817, i64 512
  store double %2036, ptr %2037, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %2038 = fpext float %2028 to double
  %2039 = getelementptr inbounds nuw i8, ptr %1817, i64 520
  store double %2038, ptr %2039, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %2040 = fpext float %2027 to double
  %2041 = getelementptr inbounds nuw i8, ptr %1817, i64 528
  store double %2040, ptr %2041, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %2042 = fpext float %2026 to double
  %2043 = getelementptr inbounds nuw i8, ptr %1817, i64 536
  store double %2042, ptr %2043, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %2044 = fpext float %1820 to double
  %2045 = getelementptr inbounds nuw i8, ptr %1817, i64 544
  store double %2044, ptr %2045, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %2046 = fpext float %1819 to double
  %2047 = getelementptr inbounds nuw i8, ptr %1817, i64 552
  store double %2046, ptr %2047, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %2048 = fpext float %1818 to double
  %2049 = getelementptr inbounds nuw i8, ptr %1817, i64 560
  store double %2048, ptr %2049, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %2050 = load float, ptr %1392, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %2051 = load float, ptr %1393, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %2052 = load float, ptr %1394, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %2053 = load float, ptr %1395, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %2054 = load float, ptr %1396, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %2055 = load float, ptr %1397, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %2056 = fpext float %2055 to double
  %2057 = getelementptr inbounds nuw i8, ptr %1817, i64 568
  store double %2056, ptr %2057, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %2058 = fpext float %2054 to double
  %2059 = getelementptr inbounds nuw i8, ptr %1817, i64 576
  store double %2058, ptr %2059, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %2060 = fpext float %2053 to double
  %2061 = getelementptr inbounds nuw i8, ptr %1817, i64 584
  store double %2060, ptr %2061, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %2062 = load float, ptr %1398, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %2063 = load float, ptr %1399, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %2064 = load float, ptr %1400, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %2065 = fpext float %2064 to double
  %2066 = getelementptr inbounds nuw i8, ptr %1817, i64 592
  store double %2065, ptr %2066, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %2067 = fpext float %2063 to double
  %2068 = getelementptr inbounds nuw i8, ptr %1817, i64 600
  store double %2067, ptr %2068, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %2069 = fpext float %2062 to double
  %2070 = getelementptr inbounds nuw i8, ptr %1817, i64 608
  store double %2069, ptr %2070, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %2071 = load float, ptr %1401, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %2072 = load float, ptr %1402, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %2073 = load float, ptr %1403, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %2074 = fpext float %2073 to double
  %2075 = getelementptr inbounds nuw i8, ptr %1817, i64 616
  store double %2074, ptr %2075, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %2076 = fpext float %2072 to double
  %2077 = getelementptr inbounds nuw i8, ptr %1817, i64 624
  store double %2076, ptr %2077, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %2078 = fpext float %2071 to double
  %2079 = getelementptr inbounds nuw i8, ptr %1817, i64 632
  store double %2078, ptr %2079, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %2080 = fpext float %2052 to double
  %2081 = getelementptr inbounds nuw i8, ptr %1817, i64 640
  store double %2080, ptr %2081, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %2082 = fpext float %2051 to double
  %2083 = getelementptr inbounds nuw i8, ptr %1817, i64 648
  store double %2082, ptr %2083, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %2084 = fpext float %2050 to double
  %2085 = getelementptr inbounds nuw i8, ptr %1817, i64 656
  store double %2084, ptr %2085, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %2086 = load float, ptr %1404, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %2087 = load float, ptr %1405, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %2088 = load float, ptr %1406, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %2089 = load float, ptr %1407, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %2090 = load float, ptr %1408, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %2091 = load float, ptr %1409, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %2092 = fpext float %2091 to double
  %2093 = getelementptr inbounds nuw i8, ptr %1817, i64 664
  store double %2092, ptr %2093, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %2094 = fpext float %2090 to double
  %2095 = getelementptr inbounds nuw i8, ptr %1817, i64 672
  store double %2094, ptr %2095, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %2096 = fpext float %2089 to double
  %2097 = getelementptr inbounds nuw i8, ptr %1817, i64 680
  store double %2096, ptr %2097, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %2098 = fpext float %2088 to double
  %2099 = getelementptr inbounds nuw i8, ptr %1817, i64 688
  store double %2098, ptr %2099, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %2100 = fpext float %2087 to double
  %2101 = getelementptr inbounds nuw i8, ptr %1817, i64 696
  store double %2100, ptr %2101, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %2102 = fpext float %2086 to double
  %2103 = getelementptr inbounds nuw i8, ptr %1817, i64 704
  store double %2102, ptr %2103, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %2104 = load float, ptr %1410, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %2105 = load float, ptr %1411, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %2106 = load float, ptr %1412, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %2107 = fpext float %2106 to double
  %2108 = getelementptr inbounds nuw i8, ptr %1817, i64 712
  store double %2107, ptr %2108, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %2109 = fpext float %2105 to double
  %2110 = getelementptr inbounds nuw i8, ptr %1817, i64 720
  store double %2109, ptr %2110, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %2111 = fpext float %2104 to double
  %2112 = getelementptr inbounds nuw i8, ptr %1817, i64 728
  store double %2111, ptr %2112, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %2113 = load float, ptr %1413, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %2114 = load float, ptr %1414, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %2115 = load float, ptr %1415, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %2116 = fpext float %2115 to double
  %2117 = getelementptr inbounds nuw i8, ptr %1817, i64 736
  store double %2116, ptr %2117, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %2118 = fpext float %2114 to double
  %2119 = getelementptr inbounds nuw i8, ptr %1817, i64 744
  store double %2118, ptr %2119, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %2120 = fpext float %2113 to double
  %2121 = getelementptr inbounds nuw i8, ptr %1817, i64 752
  store double %2120, ptr %2121, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %2122 = load float, ptr %1416, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %2123 = load float, ptr %1417, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %2124 = load float, ptr %1418, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %2125 = fpext float %2124 to double
  %2126 = getelementptr inbounds nuw i8, ptr %1817, i64 760
  store double %2125, ptr %2126, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %2127 = fpext float %2123 to double
  %2128 = getelementptr inbounds nuw i8, ptr %1817, i64 768
  store double %2127, ptr %2128, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %2129 = fpext float %2122 to double
  %2130 = getelementptr inbounds nuw i8, ptr %1817, i64 776
  store double %2129, ptr %2130, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %2131 = load float, ptr %1419, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %2132 = load float, ptr %1420, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %2133 = load float, ptr %1421, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %2134 = fpext float %2133 to double
  %2135 = getelementptr inbounds nuw i8, ptr %1817, i64 784
  store double %2134, ptr %2135, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %2136 = fpext float %2132 to double
  %2137 = getelementptr inbounds nuw i8, ptr %1817, i64 792
  store double %2136, ptr %2137, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %2138 = fpext float %2131 to double
  %2139 = getelementptr inbounds nuw i8, ptr %1817, i64 800
  store double %2138, ptr %2139, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %2140 = load float, ptr %1422, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %2141 = load float, ptr %1423, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %2142 = load float, ptr %1424, align 4, !tbaa !5, !alias.scope !30, !noalias !33
  %2143 = fpext float %2142 to double
  %2144 = getelementptr inbounds nuw i8, ptr %1817, i64 808
  store double %2143, ptr %2144, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %2145 = fpext float %2141 to double
  %2146 = getelementptr inbounds nuw i8, ptr %1817, i64 816
  store double %2145, ptr %2146, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  %2147 = fpext float %2140 to double
  %2148 = getelementptr inbounds nuw i8, ptr %1817, i64 824
  store double %2147, ptr %2148, align 8, !tbaa !7, !alias.scope !40, !noalias !43
  store ptr %.pn241419, ptr %1816, align 8, !alias.scope !45, !noalias !48
  %.fca.1.gep.i = getelementptr inbounds nuw i8, ptr %1816, i64 8
  store ptr %.pn241419, ptr %.fca.1.gep.i, align 8, !alias.scope !45, !noalias !48
  %.fca.2.gep.i = getelementptr inbounds nuw i8, ptr %1816, i64 16
  store i64 0, ptr %.fca.2.gep.i, align 8, !alias.scope !45, !noalias !48
  %.fca.3.0.gep.i = getelementptr inbounds nuw i8, ptr %1816, i64 24
  store i64 4, ptr %.fca.3.0.gep.i, align 8, !alias.scope !45, !noalias !48
  %.fca.3.1.gep.i = getelementptr inbounds nuw i8, ptr %1816, i64 32
  store i64 8, ptr %.fca.3.1.gep.i, align 8, !alias.scope !45, !noalias !48
  %.fca.3.2.gep.i = getelementptr inbounds nuw i8, ptr %1816, i64 40
  store i64 3, ptr %.fca.3.2.gep.i, align 8, !alias.scope !45, !noalias !48
  %.fca.4.0.gep.i = getelementptr inbounds nuw i8, ptr %1816, i64 48
  store i64 24, ptr %.fca.4.0.gep.i, align 8, !alias.scope !45, !noalias !48
  %.fca.4.1.gep.i = getelementptr inbounds nuw i8, ptr %1816, i64 56
  store i64 3, ptr %.fca.4.1.gep.i, align 8, !alias.scope !45, !noalias !48
  %.fca.4.2.gep.i = getelementptr inbounds nuw i8, ptr %1816, i64 64
  store i64 1, ptr %.fca.4.2.gep.i, align 8, !alias.scope !45, !noalias !48
  store ptr %23, ptr %1814, align 8, !alias.scope !50, !noalias !53
  %.fca.1.gep107.i = getelementptr inbounds nuw i8, ptr %1814, i64 8
  store ptr %24, ptr %.fca.1.gep107.i, align 8, !alias.scope !50, !noalias !53
  %.fca.2.gep109.i = getelementptr inbounds nuw i8, ptr %1814, i64 16
  store i64 %1528, ptr %.fca.2.gep109.i, align 8, !alias.scope !50, !noalias !53
  %.fca.3.0.gep111.i = getelementptr inbounds nuw i8, ptr %1814, i64 24
  store i64 8, ptr %.fca.3.0.gep111.i, align 8, !alias.scope !50, !noalias !53
  %.fca.4.0.gep113.i = getelementptr inbounds nuw i8, ptr %1814, i64 32
  store i64 1, ptr %.fca.4.0.gep113.i, align 8, !alias.scope !50, !noalias !53
  store ptr %"'mi7.i", ptr %calloc.i, align 8, !alias.scope !55, !noalias !58
  store ptr %1817, ptr %1813, align 8, !alias.scope !58, !noalias !55
  %".fca.1.gep117'ipg.i" = getelementptr inbounds nuw i8, ptr %calloc.i, i64 8
  %.fca.1.gep117.i = getelementptr inbounds nuw i8, ptr %1813, i64 8
  store ptr %"'mi7.i", ptr %".fca.1.gep117'ipg.i", align 8, !alias.scope !55, !noalias !58
  store ptr %1817, ptr %.fca.1.gep117.i, align 8, !alias.scope !58, !noalias !55
  %".fca.2.gep119'ipg.i" = getelementptr inbounds nuw i8, ptr %calloc.i, i64 16
  %.fca.2.gep119.i = getelementptr inbounds nuw i8, ptr %1813, i64 16
  store i64 0, ptr %".fca.2.gep119'ipg.i", align 8, !alias.scope !55, !noalias !58
  store i64 0, ptr %.fca.2.gep119.i, align 8, !alias.scope !58, !noalias !55
  %".fca.3.0.gep121'ipg.i" = getelementptr inbounds nuw i8, ptr %calloc.i, i64 24
  %.fca.3.0.gep121.i = getelementptr inbounds nuw i8, ptr %1813, i64 24
  store i64 104, ptr %".fca.3.0.gep121'ipg.i", align 8, !alias.scope !55, !noalias !58
  store i64 104, ptr %.fca.3.0.gep121.i, align 8, !alias.scope !58, !noalias !55
  %".fca.4.0.gep123'ipg.i" = getelementptr inbounds nuw i8, ptr %calloc.i, i64 32
  %.fca.4.0.gep123.i = getelementptr inbounds nuw i8, ptr %1813, i64 32
  store i64 1, ptr %".fca.4.0.gep123'ipg.i", align 8, !alias.scope !55, !noalias !58
  store i64 1, ptr %.fca.4.0.gep123.i, align 8, !alias.scope !58, !noalias !55
  %"'mi6.i" = tail call noalias nonnull ptr @_mlir_memref_to_llvm_alloc(i64 8) #12
  store i64 0, ptr %"'mi6.i", align 1
  %2149 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 8) #12
  store ptr %"'mi6.i", ptr %calloc4.i, align 8, !alias.scope !60, !noalias !63
  store ptr %2149, ptr %1815, align 8, !alias.scope !63, !noalias !60
  %".fca.1.gep127'ipg.i" = getelementptr inbounds nuw i8, ptr %calloc4.i, i64 8
  %.fca.1.gep127.i = getelementptr inbounds nuw i8, ptr %1815, i64 8
  store ptr %"'mi6.i", ptr %".fca.1.gep127'ipg.i", align 8, !alias.scope !60, !noalias !63
  store ptr %2149, ptr %.fca.1.gep127.i, align 8, !alias.scope !63, !noalias !60
  %".fca.2.gep129'ipg.i" = getelementptr inbounds nuw i8, ptr %calloc4.i, i64 16
  %.fca.2.gep129.i = getelementptr inbounds nuw i8, ptr %1815, i64 16
  store i64 0, ptr %".fca.2.gep129'ipg.i", align 8, !alias.scope !60, !noalias !63
  store i64 0, ptr %.fca.2.gep129.i, align 8, !alias.scope !63, !noalias !60
  tail call void @qnode_forward_0.quantum(ptr nonnull %1816, ptr nonnull %1814, ptr nonnull %1813, ptr nonnull %1815)
  %2150 = load double, ptr %2149, align 8, !tbaa !7, !alias.scope !65, !noalias !68
  %2151 = fpext float %1812 to double
  %2152 = fmul double %2150, %2151
  %2153 = fpext float %1811 to double
  %2154 = fadd double %2152, %2153
  %2155 = fpext float %1810 to double
  %2156 = fpext float %1809 to double
  %.inv.i = fcmp ole double %2154, 0.000000e+00
  %2157 = select i1 %.inv.i, double 0.000000e+00, double %2154
  %2158 = fcmp uno double %2154, 0.000000e+00
  %2159 = tail call double @llvm.fabs.f64(double %2154) #12
  %2160 = fneg double %2159
  %2161 = tail call double @llvm.exp.f64(double %2160)
  %2162 = fadd double %2161, 1.000000e+00
  %2163 = tail call double @llvm.log.f64(double %2162) #12
  %2164 = fadd double %2157, %2163
  %2165 = select i1 %2158, double %2154, double %2164
  %2166 = fmul double %2154, %2155
  %2167 = fsub double %2165, %2166
  %2168 = fmul double %2167, %2156
  %2169 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72) #12
  %2170 = ptrtoint ptr %2169 to i64
  %2171 = add i64 %2170, 63
  %2172 = and i64 %2171, -64
  %2173 = inttoptr i64 %2172 to ptr
  store double %2168, ptr %2173, align 64, !tbaa !7
  %2174 = load double, ptr %1098, align 64, !tbaa !7, !alias.scope !70, !noalias !73
  store double 0.000000e+00, ptr %1098, align 64, !tbaa !7, !alias.scope !70, !noalias !73
  %2175 = fmul fast double %2174, %2156
  %diffe8.i = select fast i1 %2158, double 0.000000e+00, double %2175
  %2176 = select fast i1 %2158, double %2175, double 0.000000e+00
  %2177 = fmul fast double %2175, %2155
  %2178 = fsub fast double %2176, %2177
  %2179 = fneg fast double %2161
  %2180 = fmul fast double %diffe8.i, %2179
  %2181 = fdiv fast double %2180, %2162
  %2182 = fcmp fast olt double %2154, 0.000000e+00
  %2183 = fneg fast double %2181
  %2184 = select fast i1 %2182, double %2183, double %2181
  %2185 = select fast i1 %.inv.i, double 0.000000e+00, double %diffe8.i
  %2186 = fadd fast double %2178, %2185
  %2187 = fadd fast double %2186, %2184
  %2188 = fptrunc fast double %2187 to float
  %2189 = fmul fast double %2187, %2151
  %2190 = fmul fast double %2187, %2150
  %2191 = fptrunc fast double %2190 to float
  %2192 = load double, ptr %"'mi6.i", align 8, !tbaa !7, !alias.scope !68, !noalias !65
  %2193 = fadd fast double %2189, %2192
  store double %2193, ptr %"'mi6.i", align 8, !tbaa !7, !alias.scope !68, !noalias !65
  %"'ipg757.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 8
  %"'ipg752.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 16
  %"'ipg747.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 24
  %"'ipg735.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 32
  %"'ipg730.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 40
  %"'ipg725.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 48
  %"'ipg720.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 56
  %"'ipg708.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 64
  %"'ipg703.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 72
  %"'ipg698.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 80
  %"'ipg693.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 88
  %"'ipg681.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 96
  %"'ipg676.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 104
  %"'ipg671.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 112
  %"'ipg666.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 120
  %"'ipg654.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 128
  %"'ipg649.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 136
  %"'ipg644.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 144
  %"'ipg639.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 152
  %"'ipg626.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 160
  %"'ipg621.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 168
  %"'ipg616.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 176
  %"'ipg611.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 184
  %"'ipg600.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 192
  %"'ipg595.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 200
  %"'ipg590.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 208
  %"'ipg585.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 216
  %"'ipg572.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 224
  %"'ipg567.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 232
  %"'ipg562.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 240
  %"'ipg557.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 248
  %"'ipg552.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 256
  %"'ipg547.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 264
  %"'ipg542.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 272
  %"'ipg528.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 280
  %"'ipg523.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 288
  %"'ipg518.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 296
  %"'ipg504.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 304
  %"'ipg499.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 312
  %"'ipg494.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 320
  %"'ipg480.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 328
  %"'ipg475.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 336
  %"'ipg470.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 344
  %"'ipg465.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 352
  %"'ipg460.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 360
  %"'ipg455.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 368
  %"'ipg432.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 376
  %"'ipg427.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 384
  %"'ipg422.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 392
  %"'ipg408.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 400
  %"'ipg403.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 408
  %"'ipg398.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 416
  %"'ipg384.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 424
  %"'ipg379.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 432
  %"'ipg374.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 440
  %"'ipg369.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 448
  %"'ipg364.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 456
  %"'ipg359.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 464
  %"'ipg345.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 472
  %"'ipg340.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 480
  %"'ipg335.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 488
  %"'ipg312.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 496
  %"'ipg307.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 504
  %"'ipg302.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 512
  %"'ipg297.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 520
  %"'ipg292.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 528
  %"'ipg287.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 536
  %"'ipg282.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 544
  %"'ipg277.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 552
  %"'ipg272.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 560
  %"'ipg249.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 568
  %"'ipg244.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 576
  %"'ipg239.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 584
  %"'ipg225.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 592
  %"'ipg220.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 600
  %"'ipg215.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 608
  %"'ipg201.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 616
  %"'ipg196.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 624
  %"'ipg191.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 632
  %"'ipg186.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 640
  %"'ipg181.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 648
  %"'ipg176.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 656
  %"'ipg153.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 664
  %"'ipg148.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 672
  %"'ipg143.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 680
  %"'ipg138.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 688
  %"'ipg133.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 696
  %"'ipg128.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 704
  %"'ipg114.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 712
  %"'ipg109.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 720
  %"'ipg104.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 728
  %"'ipg90.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 736
  %"'ipg85.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 744
  %"'ipg80.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 752
  %"'ipg66.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 760
  %"'ipg61.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 768
  %"'ipg56.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 776
  %"'ipg42.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 784
  %"'ipg37.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 792
  %"'ipg32.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 800
  %"'ipg18.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 808
  %"'ipg13.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 816
  %"'ipg.i" = getelementptr inbounds nuw i8, ptr %"'mi7.i", i64 824
  tail call void @qnode_forward_0.quantum.customqgrad(ptr nonnull %1816, ptr nonnull poison, ptr nonnull %1814, ptr nonnull poison, ptr nonnull poison, ptr nonnull %calloc.i, ptr nonnull poison, ptr nonnull %calloc4.i, ptr poison)
  tail call void @_mlir_memref_to_llvm_free(ptr nonnull %"'mi6.i")
  tail call void @_mlir_memref_to_llvm_free(ptr nonnull %2149)
  %2194 = load double, ptr %"'ipg.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2195 = fptrunc fast double %2194 to float
  %2196 = load double, ptr %"'ipg13.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg13.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2197 = fptrunc fast double %2196 to float
  %2198 = load double, ptr %"'ipg18.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg18.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2199 = fptrunc fast double %2198 to float
  %2200 = load float, ptr %1212, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2201 = fadd fast float %2200, %2199
  store float %2201, ptr %1212, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2202 = load float, ptr %1213, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2203 = fadd fast float %2202, %2197
  store float %2203, ptr %1213, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2204 = load float, ptr %1214, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2205 = fadd fast float %2204, %2195
  store float %2205, ptr %1214, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2206 = load double, ptr %"'ipg32.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg32.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2207 = fptrunc fast double %2206 to float
  %2208 = load double, ptr %"'ipg37.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg37.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2209 = fptrunc fast double %2208 to float
  %2210 = load double, ptr %"'ipg42.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg42.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2211 = fptrunc fast double %2210 to float
  %2212 = load float, ptr %1200, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2213 = fadd fast float %2212, %2211
  store float %2213, ptr %1200, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2214 = load float, ptr %1201, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2215 = fadd fast float %2214, %2209
  store float %2215, ptr %1201, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2216 = load float, ptr %1202, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2217 = fadd fast float %2216, %2207
  store float %2217, ptr %1202, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2218 = load double, ptr %"'ipg56.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg56.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2219 = fptrunc fast double %2218 to float
  %2220 = load double, ptr %"'ipg61.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg61.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2221 = fptrunc fast double %2220 to float
  %2222 = load double, ptr %"'ipg66.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg66.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2223 = fptrunc fast double %2222 to float
  %2224 = load float, ptr %1209, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2225 = fadd fast float %2224, %2223
  store float %2225, ptr %1209, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2226 = load float, ptr %1210, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2227 = fadd fast float %2226, %2221
  store float %2227, ptr %1210, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2228 = load float, ptr %1211, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2229 = fadd fast float %2228, %2219
  store float %2229, ptr %1211, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2230 = load double, ptr %"'ipg80.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg80.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2231 = fptrunc fast double %2230 to float
  %2232 = load double, ptr %"'ipg85.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg85.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2233 = fptrunc fast double %2232 to float
  %2234 = load double, ptr %"'ipg90.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg90.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2235 = fptrunc fast double %2234 to float
  %2236 = load float, ptr %1197, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2237 = fadd fast float %2236, %2235
  store float %2237, ptr %1197, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2238 = load float, ptr %1198, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2239 = fadd fast float %2238, %2233
  store float %2239, ptr %1198, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2240 = load float, ptr %1199, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2241 = fadd fast float %2240, %2231
  store float %2241, ptr %1199, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2242 = load double, ptr %"'ipg104.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg104.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2243 = fptrunc fast double %2242 to float
  %2244 = load double, ptr %"'ipg109.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg109.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2245 = fptrunc fast double %2244 to float
  %2246 = load double, ptr %"'ipg114.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg114.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2247 = fptrunc fast double %2246 to float
  %2248 = load float, ptr %1206, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2249 = fadd fast float %2248, %2247
  store float %2249, ptr %1206, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2250 = load float, ptr %1207, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2251 = fadd fast float %2250, %2245
  store float %2251, ptr %1207, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2252 = load float, ptr %1208, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2253 = fadd fast float %2252, %2243
  store float %2253, ptr %1208, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2254 = load double, ptr %"'ipg128.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg128.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2255 = fptrunc fast double %2254 to float
  %2256 = load double, ptr %"'ipg133.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg133.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2257 = fptrunc fast double %2256 to float
  %2258 = load double, ptr %"'ipg138.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg138.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2259 = fptrunc fast double %2258 to float
  %2260 = load double, ptr %"'ipg143.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg143.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2261 = fptrunc fast double %2260 to float
  %2262 = load double, ptr %"'ipg148.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg148.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2263 = fptrunc fast double %2262 to float
  %2264 = load double, ptr %"'ipg153.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg153.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2265 = fptrunc fast double %2264 to float
  %2266 = load float, ptr %1185, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2267 = fadd fast float %2266, %2265
  store float %2267, ptr %1185, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2268 = load float, ptr %1186, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2269 = fadd fast float %2268, %2263
  store float %2269, ptr %1186, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2270 = load float, ptr %1187, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2271 = fadd fast float %2270, %2261
  store float %2271, ptr %1187, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2272 = load float, ptr %1194, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2273 = fadd fast float %2272, %2259
  store float %2273, ptr %1194, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2274 = load float, ptr %1195, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2275 = fadd fast float %2274, %2257
  store float %2275, ptr %1195, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2276 = load float, ptr %1196, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2277 = fadd fast float %2276, %2255
  store float %2277, ptr %1196, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2278 = load double, ptr %"'ipg176.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg176.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2279 = fptrunc fast double %2278 to float
  %2280 = load double, ptr %"'ipg181.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg181.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2281 = fptrunc fast double %2280 to float
  %2282 = load double, ptr %"'ipg186.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg186.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2283 = fptrunc fast double %2282 to float
  %2284 = load double, ptr %"'ipg191.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg191.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2285 = fptrunc fast double %2284 to float
  %2286 = load double, ptr %"'ipg196.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg196.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2287 = fptrunc fast double %2286 to float
  %2288 = load double, ptr %"'ipg201.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg201.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2289 = fptrunc fast double %2288 to float
  %2290 = load float, ptr %1179, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2291 = fadd fast float %2290, %2289
  store float %2291, ptr %1179, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2292 = load float, ptr %1180, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2293 = fadd fast float %2292, %2287
  store float %2293, ptr %1180, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2294 = load float, ptr %1181, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2295 = fadd fast float %2294, %2285
  store float %2295, ptr %1181, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2296 = load double, ptr %"'ipg215.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg215.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2297 = fptrunc fast double %2296 to float
  %2298 = load double, ptr %"'ipg220.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg220.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2299 = fptrunc fast double %2298 to float
  %2300 = load double, ptr %"'ipg225.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg225.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2301 = fptrunc fast double %2300 to float
  %2302 = load float, ptr %1170, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2303 = fadd fast float %2302, %2301
  store float %2303, ptr %1170, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2304 = load float, ptr %1171, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2305 = fadd fast float %2304, %2299
  store float %2305, ptr %1171, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2306 = load float, ptr %1172, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2307 = fadd fast float %2306, %2297
  store float %2307, ptr %1172, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2308 = load double, ptr %"'ipg239.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg239.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2309 = fptrunc fast double %2308 to float
  %2310 = load double, ptr %"'ipg244.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg244.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2311 = fptrunc fast double %2310 to float
  %2312 = load double, ptr %"'ipg249.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg249.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2313 = fptrunc fast double %2312 to float
  %2314 = load float, ptr %1188, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2315 = fadd fast float %2314, %2313
  store float %2315, ptr %1188, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2316 = load float, ptr %1189, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2317 = fadd fast float %2316, %2311
  store float %2317, ptr %1189, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2318 = load float, ptr %1190, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2319 = fadd fast float %2318, %2309
  store float %2319, ptr %1190, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2320 = load float, ptr %1203, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2321 = fadd fast float %2320, %2283
  store float %2321, ptr %1203, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2322 = load float, ptr %1204, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2323 = fadd fast float %2322, %2281
  store float %2323, ptr %1204, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2324 = load float, ptr %1205, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2325 = fadd fast float %2324, %2279
  store float %2325, ptr %1205, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2326 = load double, ptr %"'ipg272.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg272.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2327 = fptrunc fast double %2326 to float
  %2328 = load double, ptr %"'ipg277.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg277.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2329 = fptrunc fast double %2328 to float
  %2330 = load double, ptr %"'ipg282.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg282.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2331 = fptrunc fast double %2330 to float
  %2332 = load double, ptr %"'ipg287.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg287.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2333 = fptrunc fast double %2332 to float
  %2334 = load double, ptr %"'ipg292.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg292.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2335 = fptrunc fast double %2334 to float
  %2336 = load double, ptr %"'ipg297.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg297.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2337 = fptrunc fast double %2336 to float
  %2338 = load double, ptr %"'ipg302.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg302.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2339 = fptrunc fast double %2338 to float
  %2340 = load double, ptr %"'ipg307.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg307.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2341 = fptrunc fast double %2340 to float
  %2342 = load double, ptr %"'ipg312.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg312.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2343 = fptrunc fast double %2342 to float
  %2344 = load float, ptr %1164, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2345 = fadd fast float %2344, %2343
  store float %2345, ptr %1164, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2346 = load float, ptr %1165, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2347 = fadd fast float %2346, %2341
  store float %2347, ptr %1165, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2348 = load float, ptr %1166, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2349 = fadd fast float %2348, %2339
  store float %2349, ptr %1166, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2350 = load float, ptr %1182, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2351 = fadd fast float %2350, %2337
  store float %2351, ptr %1182, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2352 = load float, ptr %1183, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2353 = fadd fast float %2352, %2335
  store float %2353, ptr %1183, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2354 = load float, ptr %1184, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2355 = fadd fast float %2354, %2333
  store float %2355, ptr %1184, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2356 = load double, ptr %"'ipg335.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg335.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2357 = fptrunc fast double %2356 to float
  %2358 = load double, ptr %"'ipg340.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg340.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2359 = fptrunc fast double %2358 to float
  %2360 = load double, ptr %"'ipg345.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg345.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2361 = fptrunc fast double %2360 to float
  %2362 = load float, ptr %1173, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2363 = fadd fast float %2362, %2361
  store float %2363, ptr %1173, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2364 = load float, ptr %1174, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2365 = fadd fast float %2364, %2359
  store float %2365, ptr %1174, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2366 = load float, ptr %1175, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2367 = fadd fast float %2366, %2357
  store float %2367, ptr %1175, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2368 = load double, ptr %"'ipg359.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg359.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2369 = fptrunc fast double %2368 to float
  %2370 = load double, ptr %"'ipg364.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg364.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2371 = fptrunc fast double %2370 to float
  %2372 = load double, ptr %"'ipg369.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg369.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2373 = fptrunc fast double %2372 to float
  %2374 = load double, ptr %"'ipg374.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg374.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2375 = fptrunc fast double %2374 to float
  %2376 = load double, ptr %"'ipg379.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg379.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2377 = fptrunc fast double %2376 to float
  %2378 = load double, ptr %"'ipg384.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg384.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2379 = fptrunc fast double %2378 to float
  %2380 = load float, ptr %1152, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2381 = fadd fast float %2380, %2379
  store float %2381, ptr %1152, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2382 = load float, ptr %1153, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2383 = fadd fast float %2382, %2377
  store float %2383, ptr %1153, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2384 = load float, ptr %1154, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2385 = fadd fast float %2384, %2375
  store float %2385, ptr %1154, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2386 = load double, ptr %"'ipg398.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg398.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2387 = fptrunc fast double %2386 to float
  %2388 = load double, ptr %"'ipg403.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg403.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2389 = fptrunc fast double %2388 to float
  %2390 = load double, ptr %"'ipg408.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg408.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2391 = fptrunc fast double %2390 to float
  %2392 = load float, ptr %1146, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2393 = fadd fast float %2392, %2391
  store float %2393, ptr %1146, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2394 = load float, ptr %1147, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2395 = fadd fast float %2394, %2389
  store float %2395, ptr %1147, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2396 = load float, ptr %1148, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2397 = fadd fast float %2396, %2387
  store float %2397, ptr %1148, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2398 = load double, ptr %"'ipg422.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg422.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2399 = fptrunc fast double %2398 to float
  %2400 = load double, ptr %"'ipg427.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg427.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2401 = fptrunc fast double %2400 to float
  %2402 = load double, ptr %"'ipg432.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg432.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2403 = fptrunc fast double %2402 to float
  %2404 = load float, ptr %1158, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2405 = fadd fast float %2404, %2403
  store float %2405, ptr %1158, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2406 = load float, ptr %1159, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2407 = fadd fast float %2406, %2401
  store float %2407, ptr %1159, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2408 = load float, ptr %1160, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2409 = fadd fast float %2408, %2399
  store float %2409, ptr %1160, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2410 = load float, ptr %1176, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2411 = fadd fast float %2410, %2373
  store float %2411, ptr %1176, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2412 = load float, ptr %1177, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2413 = fadd fast float %2412, %2371
  store float %2413, ptr %1177, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2414 = load float, ptr %1178, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2415 = fadd fast float %2414, %2369
  store float %2415, ptr %1178, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2416 = load double, ptr %"'ipg455.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg455.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2417 = fptrunc fast double %2416 to float
  %2418 = load double, ptr %"'ipg460.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg460.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2419 = fptrunc fast double %2418 to float
  %2420 = load double, ptr %"'ipg465.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg465.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2421 = fptrunc fast double %2420 to float
  %2422 = load double, ptr %"'ipg470.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg470.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2423 = fptrunc fast double %2422 to float
  %2424 = load double, ptr %"'ipg475.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg475.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2425 = fptrunc fast double %2424 to float
  %2426 = load double, ptr %"'ipg480.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg480.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2427 = fptrunc fast double %2426 to float
  %2428 = load float, ptr %1149, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2429 = fadd fast float %2428, %2427
  store float %2429, ptr %1149, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2430 = load float, ptr %1150, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2431 = fadd fast float %2430, %2425
  store float %2431, ptr %1150, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2432 = load float, ptr %1151, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2433 = fadd fast float %2432, %2423
  store float %2433, ptr %1151, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2434 = load double, ptr %"'ipg494.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg494.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2435 = fptrunc fast double %2434 to float
  %2436 = load double, ptr %"'ipg499.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg499.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2437 = fptrunc fast double %2436 to float
  %2438 = load double, ptr %"'ipg504.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg504.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2439 = fptrunc fast double %2438 to float
  %2440 = load float, ptr %1143, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2441 = fadd fast float %2440, %2439
  store float %2441, ptr %1143, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2442 = load float, ptr %1144, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2443 = fadd fast float %2442, %2437
  store float %2443, ptr %1144, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2444 = load float, ptr %1145, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2445 = fadd fast float %2444, %2435
  store float %2445, ptr %1145, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2446 = load double, ptr %"'ipg518.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg518.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2447 = fptrunc fast double %2446 to float
  %2448 = load double, ptr %"'ipg523.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg523.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2449 = fptrunc fast double %2448 to float
  %2450 = load double, ptr %"'ipg528.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg528.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2451 = fptrunc fast double %2450 to float
  %2452 = load float, ptr %1155, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2453 = fadd fast float %2452, %2451
  store float %2453, ptr %1155, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2454 = load float, ptr %1156, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2455 = fadd fast float %2454, %2449
  store float %2455, ptr %1156, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2456 = load float, ptr %1157, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2457 = fadd fast float %2456, %2447
  store float %2457, ptr %1157, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2458 = load double, ptr %"'ipg542.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg542.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2459 = fptrunc fast double %2458 to float
  %2460 = load double, ptr %"'ipg547.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg547.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2461 = fptrunc fast double %2460 to float
  %2462 = load double, ptr %"'ipg552.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg552.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2463 = fptrunc fast double %2462 to float
  %2464 = load double, ptr %"'ipg557.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg557.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2465 = fptrunc fast double %2464 to float
  %2466 = load double, ptr %"'ipg562.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg562.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2467 = fptrunc fast double %2466 to float
  %2468 = load double, ptr %"'ipg567.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2469 = fptrunc fast double %2468 to float
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %"'ipg572.i", i8 0, i64 16, i1 false)
  %2470 = load float, ptr %1122, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2471 = fadd fast float %2470, %2469
  store float %2471, ptr %1122, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2472 = load float, ptr %1123, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2473 = fadd fast float %2472, %2467
  store float %2473, ptr %1123, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2474 = load float, ptr %1124, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2475 = fadd fast float %2474, %2465
  store float %2475, ptr %1124, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2476 = load double, ptr %"'ipg585.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg585.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2477 = fptrunc fast double %2476 to float
  %2478 = load double, ptr %"'ipg590.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg590.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2479 = fptrunc fast double %2478 to float
  %2480 = load double, ptr %"'ipg595.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2481 = fptrunc fast double %2480 to float
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %"'ipg600.i", i8 0, i64 16, i1 false)
  %2482 = load float, ptr %1099, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2483 = fadd fast float %2482, %2481
  store float %2483, ptr %1099, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2484 = load float, ptr %1120, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2485 = fadd fast float %2484, %2479
  store float %2485, ptr %1120, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2486 = load float, ptr %1121, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2487 = fadd fast float %2486, %2477
  store float %2487, ptr %1121, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2488 = load double, ptr %"'ipg611.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg611.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2489 = fptrunc fast double %2488 to float
  %2490 = load double, ptr %"'ipg616.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg616.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2491 = fptrunc fast double %2490 to float
  %2492 = load double, ptr %"'ipg621.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2493 = fptrunc fast double %2492 to float
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %"'ipg626.i", i8 0, i64 16, i1 false)
  %2494 = load float, ptr %1125, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2495 = fadd fast float %2494, %2493
  store float %2495, ptr %1125, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2496 = load float, ptr %1126, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2497 = fadd fast float %2496, %2491
  store float %2497, ptr %1126, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2498 = load float, ptr %1127, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2499 = fadd fast float %2498, %2489
  store float %2499, ptr %1127, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2500 = load double, ptr %"'ipg639.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg639.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2501 = fptrunc fast double %2500 to float
  %2502 = load double, ptr %"'ipg644.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg644.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2503 = fptrunc fast double %2502 to float
  %2504 = load double, ptr %"'ipg649.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2505 = fptrunc fast double %2504 to float
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %"'ipg654.i", i8 0, i64 16, i1 false)
  %2506 = load float, ptr %1128, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2507 = fadd fast float %2506, %2505
  store float %2507, ptr %1128, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2508 = load float, ptr %1129, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2509 = fadd fast float %2508, %2503
  store float %2509, ptr %1129, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2510 = load float, ptr %1130, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2511 = fadd fast float %2510, %2501
  store float %2511, ptr %1130, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2512 = load double, ptr %"'ipg666.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg666.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2513 = fptrunc fast double %2512 to float
  %2514 = load double, ptr %"'ipg671.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg671.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2515 = fptrunc fast double %2514 to float
  %2516 = load double, ptr %"'ipg676.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2517 = fptrunc fast double %2516 to float
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %"'ipg681.i", i8 0, i64 16, i1 false)
  %2518 = load float, ptr %1131, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2519 = fadd fast float %2518, %2517
  store float %2519, ptr %1131, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2520 = load float, ptr %1132, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2521 = fadd fast float %2520, %2515
  store float %2521, ptr %1132, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2522 = load float, ptr %1133, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2523 = fadd fast float %2522, %2513
  store float %2523, ptr %1133, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2524 = load double, ptr %"'ipg693.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg693.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2525 = fptrunc fast double %2524 to float
  %2526 = load double, ptr %"'ipg698.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg698.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2527 = fptrunc fast double %2526 to float
  %2528 = load double, ptr %"'ipg703.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2529 = fptrunc fast double %2528 to float
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %"'ipg708.i", i8 0, i64 16, i1 false)
  %2530 = load float, ptr %1134, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2531 = fadd fast float %2530, %2529
  store float %2531, ptr %1134, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2532 = load float, ptr %1135, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2533 = fadd fast float %2532, %2527
  store float %2533, ptr %1135, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2534 = load float, ptr %1136, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2535 = fadd fast float %2534, %2525
  store float %2535, ptr %1136, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2536 = load double, ptr %"'ipg720.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg720.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2537 = fptrunc fast double %2536 to float
  %2538 = load double, ptr %"'ipg725.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg725.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2539 = fptrunc fast double %2538 to float
  %2540 = load double, ptr %"'ipg730.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2541 = fptrunc fast double %2540 to float
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %"'ipg735.i", i8 0, i64 16, i1 false)
  %2542 = load float, ptr %1137, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2543 = fadd fast float %2542, %2541
  store float %2543, ptr %1137, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2544 = load float, ptr %1138, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2545 = fadd fast float %2544, %2539
  store float %2545, ptr %1138, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2546 = load float, ptr %1139, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2547 = fadd fast float %2546, %2537
  store float %2547, ptr %1139, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2548 = load double, ptr %"'ipg747.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg747.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2549 = fptrunc fast double %2548 to float
  %2550 = load double, ptr %"'ipg752.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  store double 0.000000e+00, ptr %"'ipg752.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2551 = fptrunc fast double %2550 to float
  %2552 = load double, ptr %"'ipg757.i", align 8, !tbaa !7, !alias.scope !75, !noalias !78
  %2553 = fptrunc fast double %2552 to float
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %"'mi7.i", i8 0, i64 16, i1 false)
  %2554 = load float, ptr %1140, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2555 = fadd fast float %2554, %2553
  store float %2555, ptr %1140, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2556 = load float, ptr %1141, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2557 = fadd fast float %2556, %2551
  store float %2557, ptr %1141, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2558 = load float, ptr %1142, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2559 = fadd fast float %2558, %2549
  store float %2559, ptr %1142, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2560 = load float, ptr %1161, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2561 = fadd fast float %2560, %2463
  store float %2561, ptr %1161, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2562 = load float, ptr %1162, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2563 = fadd fast float %2562, %2461
  store float %2563, ptr %1162, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2564 = load float, ptr %1163, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2565 = fadd fast float %2564, %2459
  store float %2565, ptr %1163, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2566 = load float, ptr %1167, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2567 = fadd fast float %2566, %2421
  store float %2567, ptr %1167, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2568 = load float, ptr %1168, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2569 = fadd fast float %2568, %2419
  store float %2569, ptr %1168, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2570 = load float, ptr %1169, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2571 = fadd fast float %2570, %2417
  store float %2571, ptr %1169, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2572 = load float, ptr %1191, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2573 = fadd fast float %2572, %2331
  store float %2573, ptr %1191, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2574 = load float, ptr %1192, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2575 = fadd fast float %2574, %2329
  store float %2575, ptr %1192, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2576 = load float, ptr %1193, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  %2577 = fadd fast float %2576, %2327
  store float %2577, ptr %1193, align 4, !tbaa !5, !alias.scope !80, !noalias !83
  tail call void @_mlir_memref_to_llvm_free(ptr nonnull %"'mi7.i")
  tail call void @_mlir_memref_to_llvm_free(ptr %1817)
  tail call void @free(ptr nonnull %1816)
  tail call void @free(ptr nonnull %calloc4.i)
  tail call void @free(ptr nonnull %1815)
  tail call void @free(ptr nonnull %1814)
  tail call void @free(ptr nonnull %calloc.i)
  tail call void @free(ptr nonnull %1813)
  %2578 = load float, ptr %1101, align 4, !tbaa !5, !alias.scope !27, !noalias !24
  %2579 = fadd fast float %2578, %2191
  store float %2579, ptr %1101, align 4, !tbaa !5, !alias.scope !27, !noalias !24
  %2580 = load float, ptr %1100, align 4, !tbaa !5, !alias.scope !22, !noalias !19
  %2581 = fadd fast float %2580, %2188
  store float %2581, ptr %1100, align 4, !tbaa !5, !alias.scope !22, !noalias !19
  %2582 = load float, ptr %1101, align 4, !tbaa !5
  %2583 = load float, ptr %.pn297418, align 4, !tbaa !5
  %2584 = load float, ptr %1099, align 4, !tbaa !5
  %2585 = fadd float %2583, %2584
  store float %2585, ptr %1326, align 64, !tbaa !5
  %2586 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 4
  %2587 = load float, ptr %2586, align 4, !tbaa !5
  %2588 = load float, ptr %1120, align 4, !tbaa !5
  %2589 = fadd float %2587, %2588
  store float %2589, ptr %1427, align 4, !tbaa !5
  %2590 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 8
  %2591 = load float, ptr %2590, align 4, !tbaa !5
  %2592 = load float, ptr %1121, align 4, !tbaa !5
  %2593 = fadd float %2591, %2592
  store float %2593, ptr %1428, align 8, !tbaa !5
  %2594 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 12
  %2595 = load float, ptr %2594, align 4, !tbaa !5
  %2596 = load float, ptr %1122, align 4, !tbaa !5
  %2597 = fadd float %2595, %2596
  store float %2597, ptr %1429, align 4, !tbaa !5
  %2598 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 16
  %2599 = load float, ptr %2598, align 4, !tbaa !5
  %2600 = load float, ptr %1123, align 4, !tbaa !5
  %2601 = fadd float %2599, %2600
  store float %2601, ptr %1430, align 16, !tbaa !5
  %2602 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 20
  %2603 = load float, ptr %2602, align 4, !tbaa !5
  %2604 = load float, ptr %1124, align 4, !tbaa !5
  %2605 = fadd float %2603, %2604
  store float %2605, ptr %1431, align 4, !tbaa !5
  %2606 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 24
  %2607 = load float, ptr %2606, align 4, !tbaa !5
  %2608 = load float, ptr %1125, align 4, !tbaa !5
  %2609 = fadd float %2607, %2608
  store float %2609, ptr %1432, align 8, !tbaa !5
  %2610 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 28
  %2611 = load float, ptr %2610, align 4, !tbaa !5
  %2612 = load float, ptr %1126, align 4, !tbaa !5
  %2613 = fadd float %2611, %2612
  store float %2613, ptr %1433, align 4, !tbaa !5
  %2614 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 32
  %2615 = load float, ptr %2614, align 4, !tbaa !5
  %2616 = load float, ptr %1127, align 4, !tbaa !5
  %2617 = fadd float %2615, %2616
  store float %2617, ptr %1434, align 32, !tbaa !5
  %2618 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 36
  %2619 = load float, ptr %2618, align 4, !tbaa !5
  %2620 = load float, ptr %1128, align 4, !tbaa !5
  %2621 = fadd float %2619, %2620
  store float %2621, ptr %1435, align 4, !tbaa !5
  %2622 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 40
  %2623 = load float, ptr %2622, align 4, !tbaa !5
  %2624 = load float, ptr %1129, align 4, !tbaa !5
  %2625 = fadd float %2623, %2624
  store float %2625, ptr %1436, align 8, !tbaa !5
  %2626 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 44
  %2627 = load float, ptr %2626, align 4, !tbaa !5
  %2628 = load float, ptr %1130, align 4, !tbaa !5
  %2629 = fadd float %2627, %2628
  store float %2629, ptr %1437, align 4, !tbaa !5
  %2630 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 48
  %2631 = load float, ptr %2630, align 4, !tbaa !5
  %2632 = load float, ptr %1131, align 4, !tbaa !5
  %2633 = fadd float %2631, %2632
  store float %2633, ptr %1438, align 16, !tbaa !5
  %2634 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 52
  %2635 = load float, ptr %2634, align 4, !tbaa !5
  %2636 = load float, ptr %1132, align 4, !tbaa !5
  %2637 = fadd float %2635, %2636
  store float %2637, ptr %1439, align 4, !tbaa !5
  %2638 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 56
  %2639 = load float, ptr %2638, align 4, !tbaa !5
  %2640 = load float, ptr %1133, align 4, !tbaa !5
  %2641 = fadd float %2639, %2640
  store float %2641, ptr %1440, align 8, !tbaa !5
  %2642 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 60
  %2643 = load float, ptr %2642, align 4, !tbaa !5
  %2644 = load float, ptr %1134, align 4, !tbaa !5
  %2645 = fadd float %2643, %2644
  store float %2645, ptr %1441, align 4, !tbaa !5
  %2646 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 64
  %2647 = load float, ptr %2646, align 4, !tbaa !5
  %2648 = load float, ptr %1135, align 4, !tbaa !5
  %2649 = fadd float %2647, %2648
  store float %2649, ptr %1442, align 64, !tbaa !5
  %2650 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 68
  %2651 = load float, ptr %2650, align 4, !tbaa !5
  %2652 = load float, ptr %1136, align 4, !tbaa !5
  %2653 = fadd float %2651, %2652
  store float %2653, ptr %1443, align 4, !tbaa !5
  %2654 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 72
  %2655 = load float, ptr %2654, align 4, !tbaa !5
  %2656 = load float, ptr %1137, align 4, !tbaa !5
  %2657 = fadd float %2655, %2656
  store float %2657, ptr %1444, align 8, !tbaa !5
  %2658 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 76
  %2659 = load float, ptr %2658, align 4, !tbaa !5
  %2660 = load float, ptr %1138, align 4, !tbaa !5
  %2661 = fadd float %2659, %2660
  store float %2661, ptr %1445, align 4, !tbaa !5
  %2662 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 80
  %2663 = load float, ptr %2662, align 4, !tbaa !5
  %2664 = load float, ptr %1139, align 4, !tbaa !5
  %2665 = fadd float %2663, %2664
  store float %2665, ptr %1446, align 16, !tbaa !5
  %2666 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 84
  %2667 = load float, ptr %2666, align 4, !tbaa !5
  %2668 = load float, ptr %1140, align 4, !tbaa !5
  %2669 = fadd float %2667, %2668
  store float %2669, ptr %1447, align 4, !tbaa !5
  %2670 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 88
  %2671 = load float, ptr %2670, align 4, !tbaa !5
  %2672 = load float, ptr %1141, align 4, !tbaa !5
  %2673 = fadd float %2671, %2672
  store float %2673, ptr %1448, align 8, !tbaa !5
  %2674 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 92
  %2675 = load float, ptr %2674, align 4, !tbaa !5
  %2676 = load float, ptr %1142, align 4, !tbaa !5
  %2677 = fadd float %2675, %2676
  store float %2677, ptr %1449, align 4, !tbaa !5
  %2678 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 96
  %2679 = load float, ptr %2678, align 4, !tbaa !5
  %2680 = load float, ptr %1143, align 4, !tbaa !5
  %2681 = fadd float %2679, %2680
  store float %2681, ptr %1450, align 32, !tbaa !5
  %2682 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 100
  %2683 = load float, ptr %2682, align 4, !tbaa !5
  %2684 = load float, ptr %1144, align 4, !tbaa !5
  %2685 = fadd float %2683, %2684
  store float %2685, ptr %1451, align 4, !tbaa !5
  %2686 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 104
  %2687 = load float, ptr %2686, align 4, !tbaa !5
  %2688 = load float, ptr %1145, align 4, !tbaa !5
  %2689 = fadd float %2687, %2688
  store float %2689, ptr %1452, align 8, !tbaa !5
  %2690 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 108
  %2691 = load float, ptr %2690, align 4, !tbaa !5
  %2692 = load float, ptr %1146, align 4, !tbaa !5
  %2693 = fadd float %2691, %2692
  store float %2693, ptr %1453, align 4, !tbaa !5
  %2694 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 112
  %2695 = load float, ptr %2694, align 4, !tbaa !5
  %2696 = load float, ptr %1147, align 4, !tbaa !5
  %2697 = fadd float %2695, %2696
  store float %2697, ptr %1454, align 16, !tbaa !5
  %2698 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 116
  %2699 = load float, ptr %2698, align 4, !tbaa !5
  %2700 = load float, ptr %1148, align 4, !tbaa !5
  %2701 = fadd float %2699, %2700
  store float %2701, ptr %1455, align 4, !tbaa !5
  %2702 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 120
  %2703 = load float, ptr %2702, align 4, !tbaa !5
  %2704 = load float, ptr %1149, align 4, !tbaa !5
  %2705 = fadd float %2703, %2704
  store float %2705, ptr %1456, align 8, !tbaa !5
  %2706 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 124
  %2707 = load float, ptr %2706, align 4, !tbaa !5
  %2708 = load float, ptr %1150, align 4, !tbaa !5
  %2709 = fadd float %2707, %2708
  store float %2709, ptr %1457, align 4, !tbaa !5
  %2710 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 128
  %2711 = load float, ptr %2710, align 4, !tbaa !5
  %2712 = load float, ptr %1151, align 4, !tbaa !5
  %2713 = fadd float %2711, %2712
  store float %2713, ptr %1458, align 64, !tbaa !5
  %2714 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 132
  %2715 = load float, ptr %2714, align 4, !tbaa !5
  %2716 = load float, ptr %1152, align 4, !tbaa !5
  %2717 = fadd float %2715, %2716
  store float %2717, ptr %1459, align 4, !tbaa !5
  %2718 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 136
  %2719 = load float, ptr %2718, align 4, !tbaa !5
  %2720 = load float, ptr %1153, align 4, !tbaa !5
  %2721 = fadd float %2719, %2720
  store float %2721, ptr %1460, align 8, !tbaa !5
  %2722 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 140
  %2723 = load float, ptr %2722, align 4, !tbaa !5
  %2724 = load float, ptr %1154, align 4, !tbaa !5
  %2725 = fadd float %2723, %2724
  store float %2725, ptr %1461, align 4, !tbaa !5
  %2726 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 144
  %2727 = load float, ptr %2726, align 4, !tbaa !5
  %2728 = load float, ptr %1155, align 4, !tbaa !5
  %2729 = fadd float %2727, %2728
  store float %2729, ptr %1462, align 16, !tbaa !5
  %2730 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 148
  %2731 = load float, ptr %2730, align 4, !tbaa !5
  %2732 = load float, ptr %1156, align 4, !tbaa !5
  %2733 = fadd float %2731, %2732
  store float %2733, ptr %1463, align 4, !tbaa !5
  %2734 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 152
  %2735 = load float, ptr %2734, align 4, !tbaa !5
  %2736 = load float, ptr %1157, align 4, !tbaa !5
  %2737 = fadd float %2735, %2736
  store float %2737, ptr %1464, align 8, !tbaa !5
  %2738 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 156
  %2739 = load float, ptr %2738, align 4, !tbaa !5
  %2740 = load float, ptr %1158, align 4, !tbaa !5
  %2741 = fadd float %2739, %2740
  store float %2741, ptr %1465, align 4, !tbaa !5
  %2742 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 160
  %2743 = load float, ptr %2742, align 4, !tbaa !5
  %2744 = load float, ptr %1159, align 4, !tbaa !5
  %2745 = fadd float %2743, %2744
  store float %2745, ptr %1466, align 32, !tbaa !5
  %2746 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 164
  %2747 = load float, ptr %2746, align 4, !tbaa !5
  %2748 = load float, ptr %1160, align 4, !tbaa !5
  %2749 = fadd float %2747, %2748
  store float %2749, ptr %1467, align 4, !tbaa !5
  %2750 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 168
  %2751 = load float, ptr %2750, align 4, !tbaa !5
  %2752 = load float, ptr %1161, align 4, !tbaa !5
  %2753 = fadd float %2751, %2752
  store float %2753, ptr %1468, align 8, !tbaa !5
  %2754 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 172
  %2755 = load float, ptr %2754, align 4, !tbaa !5
  %2756 = load float, ptr %1162, align 4, !tbaa !5
  %2757 = fadd float %2755, %2756
  store float %2757, ptr %1469, align 4, !tbaa !5
  %2758 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 176
  %2759 = load float, ptr %2758, align 4, !tbaa !5
  %2760 = load float, ptr %1163, align 4, !tbaa !5
  %2761 = fadd float %2759, %2760
  store float %2761, ptr %1470, align 16, !tbaa !5
  %2762 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 180
  %2763 = load float, ptr %2762, align 4, !tbaa !5
  %2764 = load float, ptr %1164, align 4, !tbaa !5
  %2765 = fadd float %2763, %2764
  store float %2765, ptr %1471, align 4, !tbaa !5
  %2766 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 184
  %2767 = load float, ptr %2766, align 4, !tbaa !5
  %2768 = load float, ptr %1165, align 4, !tbaa !5
  %2769 = fadd float %2767, %2768
  store float %2769, ptr %1472, align 8, !tbaa !5
  %2770 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 188
  %2771 = load float, ptr %2770, align 4, !tbaa !5
  %2772 = load float, ptr %1166, align 4, !tbaa !5
  %2773 = fadd float %2771, %2772
  store float %2773, ptr %1473, align 4, !tbaa !5
  %2774 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 192
  %2775 = load float, ptr %2774, align 4, !tbaa !5
  %2776 = load float, ptr %1167, align 4, !tbaa !5
  %2777 = fadd float %2775, %2776
  store float %2777, ptr %1474, align 64, !tbaa !5
  %2778 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 196
  %2779 = load float, ptr %2778, align 4, !tbaa !5
  %2780 = load float, ptr %1168, align 4, !tbaa !5
  %2781 = fadd float %2779, %2780
  store float %2781, ptr %1475, align 4, !tbaa !5
  %2782 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 200
  %2783 = load float, ptr %2782, align 4, !tbaa !5
  %2784 = load float, ptr %1169, align 4, !tbaa !5
  %2785 = fadd float %2783, %2784
  store float %2785, ptr %1476, align 8, !tbaa !5
  %2786 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 204
  %2787 = load float, ptr %2786, align 4, !tbaa !5
  %2788 = load float, ptr %1170, align 4, !tbaa !5
  %2789 = fadd float %2787, %2788
  store float %2789, ptr %1477, align 4, !tbaa !5
  %2790 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 208
  %2791 = load float, ptr %2790, align 4, !tbaa !5
  %2792 = load float, ptr %1171, align 4, !tbaa !5
  %2793 = fadd float %2791, %2792
  store float %2793, ptr %1478, align 16, !tbaa !5
  %2794 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 212
  %2795 = load float, ptr %2794, align 4, !tbaa !5
  %2796 = load float, ptr %1172, align 4, !tbaa !5
  %2797 = fadd float %2795, %2796
  store float %2797, ptr %1479, align 4, !tbaa !5
  %2798 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 216
  %2799 = load float, ptr %2798, align 4, !tbaa !5
  %2800 = load float, ptr %1173, align 4, !tbaa !5
  %2801 = fadd float %2799, %2800
  store float %2801, ptr %1480, align 8, !tbaa !5
  %2802 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 220
  %2803 = load float, ptr %2802, align 4, !tbaa !5
  %2804 = load float, ptr %1174, align 4, !tbaa !5
  %2805 = fadd float %2803, %2804
  store float %2805, ptr %1481, align 4, !tbaa !5
  %2806 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 224
  %2807 = load float, ptr %2806, align 4, !tbaa !5
  %2808 = load float, ptr %1175, align 4, !tbaa !5
  %2809 = fadd float %2807, %2808
  store float %2809, ptr %1482, align 32, !tbaa !5
  %2810 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 228
  %2811 = load float, ptr %2810, align 4, !tbaa !5
  %2812 = load float, ptr %1176, align 4, !tbaa !5
  %2813 = fadd float %2811, %2812
  store float %2813, ptr %1483, align 4, !tbaa !5
  %2814 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 232
  %2815 = load float, ptr %2814, align 4, !tbaa !5
  %2816 = load float, ptr %1177, align 4, !tbaa !5
  %2817 = fadd float %2815, %2816
  store float %2817, ptr %1484, align 8, !tbaa !5
  %2818 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 236
  %2819 = load float, ptr %2818, align 4, !tbaa !5
  %2820 = load float, ptr %1178, align 4, !tbaa !5
  %2821 = fadd float %2819, %2820
  store float %2821, ptr %1485, align 4, !tbaa !5
  %2822 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 240
  %2823 = load float, ptr %2822, align 4, !tbaa !5
  %2824 = load float, ptr %1179, align 4, !tbaa !5
  %2825 = fadd float %2823, %2824
  store float %2825, ptr %1486, align 16, !tbaa !5
  %2826 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 244
  %2827 = load float, ptr %2826, align 4, !tbaa !5
  %2828 = load float, ptr %1180, align 4, !tbaa !5
  %2829 = fadd float %2827, %2828
  store float %2829, ptr %1487, align 4, !tbaa !5
  %2830 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 248
  %2831 = load float, ptr %2830, align 4, !tbaa !5
  %2832 = load float, ptr %1181, align 4, !tbaa !5
  %2833 = fadd float %2831, %2832
  store float %2833, ptr %1488, align 8, !tbaa !5
  %2834 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 252
  %2835 = load float, ptr %2834, align 4, !tbaa !5
  %2836 = load float, ptr %1182, align 4, !tbaa !5
  %2837 = fadd float %2835, %2836
  store float %2837, ptr %1489, align 4, !tbaa !5
  %2838 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 256
  %2839 = load float, ptr %2838, align 4, !tbaa !5
  %2840 = load float, ptr %1183, align 4, !tbaa !5
  %2841 = fadd float %2839, %2840
  store float %2841, ptr %1490, align 64, !tbaa !5
  %2842 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 260
  %2843 = load float, ptr %2842, align 4, !tbaa !5
  %2844 = load float, ptr %1184, align 4, !tbaa !5
  %2845 = fadd float %2843, %2844
  store float %2845, ptr %1491, align 4, !tbaa !5
  %2846 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 264
  %2847 = load float, ptr %2846, align 4, !tbaa !5
  %2848 = load float, ptr %1185, align 4, !tbaa !5
  %2849 = fadd float %2847, %2848
  store float %2849, ptr %1492, align 8, !tbaa !5
  %2850 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 268
  %2851 = load float, ptr %2850, align 4, !tbaa !5
  %2852 = load float, ptr %1186, align 4, !tbaa !5
  %2853 = fadd float %2851, %2852
  store float %2853, ptr %1493, align 4, !tbaa !5
  %2854 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 272
  %2855 = load float, ptr %2854, align 4, !tbaa !5
  %2856 = load float, ptr %1187, align 4, !tbaa !5
  %2857 = fadd float %2855, %2856
  store float %2857, ptr %1494, align 16, !tbaa !5
  %2858 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 276
  %2859 = load float, ptr %2858, align 4, !tbaa !5
  %2860 = load float, ptr %1188, align 4, !tbaa !5
  %2861 = fadd float %2859, %2860
  store float %2861, ptr %1495, align 4, !tbaa !5
  %2862 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 280
  %2863 = load float, ptr %2862, align 4, !tbaa !5
  %2864 = load float, ptr %1189, align 4, !tbaa !5
  %2865 = fadd float %2863, %2864
  store float %2865, ptr %1496, align 8, !tbaa !5
  %2866 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 284
  %2867 = load float, ptr %2866, align 4, !tbaa !5
  %2868 = load float, ptr %1190, align 4, !tbaa !5
  %2869 = fadd float %2867, %2868
  store float %2869, ptr %1497, align 4, !tbaa !5
  %2870 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 288
  %2871 = load float, ptr %2870, align 4, !tbaa !5
  %2872 = load float, ptr %1191, align 4, !tbaa !5
  %2873 = fadd float %2871, %2872
  store float %2873, ptr %1498, align 32, !tbaa !5
  %2874 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 292
  %2875 = load float, ptr %2874, align 4, !tbaa !5
  %2876 = load float, ptr %1192, align 4, !tbaa !5
  %2877 = fadd float %2875, %2876
  store float %2877, ptr %1499, align 4, !tbaa !5
  %2878 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 296
  %2879 = load float, ptr %2878, align 4, !tbaa !5
  %2880 = load float, ptr %1193, align 4, !tbaa !5
  %2881 = fadd float %2879, %2880
  store float %2881, ptr %1500, align 8, !tbaa !5
  %2882 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 300
  %2883 = load float, ptr %2882, align 4, !tbaa !5
  %2884 = load float, ptr %1194, align 4, !tbaa !5
  %2885 = fadd float %2883, %2884
  store float %2885, ptr %1501, align 4, !tbaa !5
  %2886 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 304
  %2887 = load float, ptr %2886, align 4, !tbaa !5
  %2888 = load float, ptr %1195, align 4, !tbaa !5
  %2889 = fadd float %2887, %2888
  store float %2889, ptr %1502, align 16, !tbaa !5
  %2890 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 308
  %2891 = load float, ptr %2890, align 4, !tbaa !5
  %2892 = load float, ptr %1196, align 4, !tbaa !5
  %2893 = fadd float %2891, %2892
  store float %2893, ptr %1503, align 4, !tbaa !5
  %2894 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 312
  %2895 = load float, ptr %2894, align 4, !tbaa !5
  %2896 = load float, ptr %1197, align 4, !tbaa !5
  %2897 = fadd float %2895, %2896
  store float %2897, ptr %1504, align 8, !tbaa !5
  %2898 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 316
  %2899 = load float, ptr %2898, align 4, !tbaa !5
  %2900 = load float, ptr %1198, align 4, !tbaa !5
  %2901 = fadd float %2899, %2900
  store float %2901, ptr %1505, align 4, !tbaa !5
  %2902 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 320
  %2903 = load float, ptr %2902, align 4, !tbaa !5
  %2904 = load float, ptr %1199, align 4, !tbaa !5
  %2905 = fadd float %2903, %2904
  store float %2905, ptr %1506, align 64, !tbaa !5
  %2906 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 324
  %2907 = load float, ptr %2906, align 4, !tbaa !5
  %2908 = load float, ptr %1200, align 4, !tbaa !5
  %2909 = fadd float %2907, %2908
  store float %2909, ptr %1507, align 4, !tbaa !5
  %2910 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 328
  %2911 = load float, ptr %2910, align 4, !tbaa !5
  %2912 = load float, ptr %1201, align 4, !tbaa !5
  %2913 = fadd float %2911, %2912
  store float %2913, ptr %1508, align 8, !tbaa !5
  %2914 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 332
  %2915 = load float, ptr %2914, align 4, !tbaa !5
  %2916 = load float, ptr %1202, align 4, !tbaa !5
  %2917 = fadd float %2915, %2916
  store float %2917, ptr %1509, align 4, !tbaa !5
  %2918 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 336
  %2919 = load float, ptr %2918, align 4, !tbaa !5
  %2920 = load float, ptr %1203, align 4, !tbaa !5
  %2921 = fadd float %2919, %2920
  store float %2921, ptr %1510, align 16, !tbaa !5
  %2922 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 340
  %2923 = load float, ptr %2922, align 4, !tbaa !5
  %2924 = load float, ptr %1204, align 4, !tbaa !5
  %2925 = fadd float %2923, %2924
  store float %2925, ptr %1511, align 4, !tbaa !5
  %2926 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 344
  %2927 = load float, ptr %2926, align 4, !tbaa !5
  %2928 = load float, ptr %1205, align 4, !tbaa !5
  %2929 = fadd float %2927, %2928
  store float %2929, ptr %1512, align 8, !tbaa !5
  %2930 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 348
  %2931 = load float, ptr %2930, align 4, !tbaa !5
  %2932 = load float, ptr %1206, align 4, !tbaa !5
  %2933 = fadd float %2931, %2932
  store float %2933, ptr %1513, align 4, !tbaa !5
  %2934 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 352
  %2935 = load float, ptr %2934, align 4, !tbaa !5
  %2936 = load float, ptr %1207, align 4, !tbaa !5
  %2937 = fadd float %2935, %2936
  store float %2937, ptr %1514, align 32, !tbaa !5
  %2938 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 356
  %2939 = load float, ptr %2938, align 4, !tbaa !5
  %2940 = load float, ptr %1208, align 4, !tbaa !5
  %2941 = fadd float %2939, %2940
  store float %2941, ptr %1515, align 4, !tbaa !5
  %2942 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 360
  %2943 = load float, ptr %2942, align 4, !tbaa !5
  %2944 = load float, ptr %1209, align 4, !tbaa !5
  %2945 = fadd float %2943, %2944
  store float %2945, ptr %1516, align 8, !tbaa !5
  %2946 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 364
  %2947 = load float, ptr %2946, align 4, !tbaa !5
  %2948 = load float, ptr %1210, align 4, !tbaa !5
  %2949 = fadd float %2947, %2948
  store float %2949, ptr %1517, align 4, !tbaa !5
  %2950 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 368
  %2951 = load float, ptr %2950, align 4, !tbaa !5
  %2952 = load float, ptr %1211, align 4, !tbaa !5
  %2953 = fadd float %2951, %2952
  store float %2953, ptr %1518, align 16, !tbaa !5
  %2954 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 372
  %2955 = load float, ptr %2954, align 4, !tbaa !5
  %2956 = load float, ptr %1212, align 4, !tbaa !5
  %2957 = fadd float %2955, %2956
  store float %2957, ptr %1519, align 4, !tbaa !5
  %2958 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 376
  %2959 = load float, ptr %2958, align 4, !tbaa !5
  %2960 = load float, ptr %1213, align 4, !tbaa !5
  %2961 = fadd float %2959, %2960
  store float %2961, ptr %1520, align 8, !tbaa !5
  %2962 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 380
  %2963 = load float, ptr %2962, align 4, !tbaa !5
  %2964 = load float, ptr %1214, align 4, !tbaa !5
  %2965 = fadd float %2963, %2964
  store float %2965, ptr %1521, align 4, !tbaa !5
  %2966 = fmul double %1808, %1425
  %2967 = fadd double %2966, %1426
  %2968 = fpext float %1530 to double
  %2969 = fpext float %1532 to double
  %.inv = fcmp ole double %2967, 0.000000e+00
  %2970 = select i1 %.inv, double 0.000000e+00, double %2967
  %2971 = fcmp uno double %2967, 0.000000e+00
  %2972 = tail call double @llvm.fabs.f64(double %2967)
  %2973 = fneg double %2972
  %2974 = tail call double @llvm.exp.f64(double %2973)
  %2975 = fadd double %2974, 1.000000e+00
  %2976 = tail call double @llvm.log.f64(double %2975)
  %2977 = fadd double %2970, %2976
  %2978 = select i1 %2971, double %2967, double %2977
  %2979 = fmul double %2967, %2968
  %2980 = fsub double %2978, %2979
  %2981 = fmul double %2980, %2969
  tail call void @_mlir_memref_to_llvm_free(ptr nonnull %.pn297418)
  %2982 = fpext float %2581 to double
  %2983 = fadd double %1524, %2982
  %2984 = fpext float %2582 to double
  %2985 = fadd double %1523, %2984
  %2986 = fadd double %1522, %2981
  %2987 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 448)
  %2988 = ptrtoint ptr %2987 to i64
  %2989 = add i64 %2988, 63
  %2990 = and i64 %2989, -64
  %2991 = inttoptr i64 %2990 to ptr
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 64 dereferenceable(384) %2991, ptr noundef nonnull align 64 dereferenceable(384) %1326, i64 384, i1 false)
  %2992 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 384)
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(384) %2992, ptr noundef nonnull align 64 dereferenceable(384) %2991, i64 384, i1 false)
  tail call void @_mlir_memref_to_llvm_free(ptr %2987)
  %2993 = add nuw nsw i64 %1525, 1
  %exitcond460.not = icmp eq i64 %2993, 32
  br i1 %exitcond460.not, label %.preheader338, label %.preheader332

.preheader338:                                    ; preds = %.preheader332, %.preheader338
  %2994 = phi i64 [ %3019, %.preheader338 ], [ 0, %.preheader332 ]
  %.idx301 = mul nuw nsw i64 %2994, 96
  %2995 = getelementptr i8, ptr %1326, i64 %.idx301
  store float 3.125000e-02, ptr %2995, align 32, !tbaa !5
  %2996 = getelementptr i8, ptr %2995, i64 4
  store float 3.125000e-02, ptr %2996, align 4, !tbaa !5
  %2997 = getelementptr i8, ptr %2995, i64 8
  store float 3.125000e-02, ptr %2997, align 8, !tbaa !5
  %2998 = getelementptr i8, ptr %2995, i64 12
  store float 3.125000e-02, ptr %2998, align 4, !tbaa !5
  %2999 = getelementptr i8, ptr %2995, i64 16
  store float 3.125000e-02, ptr %2999, align 16, !tbaa !5
  %3000 = getelementptr i8, ptr %2995, i64 20
  store float 3.125000e-02, ptr %3000, align 4, !tbaa !5
  %3001 = getelementptr i8, ptr %2995, i64 24
  store float 3.125000e-02, ptr %3001, align 8, !tbaa !5
  %3002 = getelementptr i8, ptr %2995, i64 28
  store float 3.125000e-02, ptr %3002, align 4, !tbaa !5
  %3003 = getelementptr i8, ptr %2995, i64 32
  store float 3.125000e-02, ptr %3003, align 32, !tbaa !5
  %3004 = getelementptr i8, ptr %2995, i64 36
  store float 3.125000e-02, ptr %3004, align 4, !tbaa !5
  %3005 = getelementptr i8, ptr %2995, i64 40
  store float 3.125000e-02, ptr %3005, align 8, !tbaa !5
  %3006 = getelementptr i8, ptr %2995, i64 44
  store float 3.125000e-02, ptr %3006, align 4, !tbaa !5
  %3007 = getelementptr i8, ptr %2995, i64 48
  store float 3.125000e-02, ptr %3007, align 16, !tbaa !5
  %3008 = getelementptr i8, ptr %2995, i64 52
  store float 3.125000e-02, ptr %3008, align 4, !tbaa !5
  %3009 = getelementptr i8, ptr %2995, i64 56
  store float 3.125000e-02, ptr %3009, align 8, !tbaa !5
  %3010 = getelementptr i8, ptr %2995, i64 60
  store float 3.125000e-02, ptr %3010, align 4, !tbaa !5
  %3011 = getelementptr i8, ptr %2995, i64 64
  store float 3.125000e-02, ptr %3011, align 32, !tbaa !5
  %3012 = getelementptr i8, ptr %2995, i64 68
  store float 3.125000e-02, ptr %3012, align 4, !tbaa !5
  %3013 = getelementptr i8, ptr %2995, i64 72
  store float 3.125000e-02, ptr %3013, align 8, !tbaa !5
  %3014 = getelementptr i8, ptr %2995, i64 76
  store float 3.125000e-02, ptr %3014, align 4, !tbaa !5
  %3015 = getelementptr i8, ptr %2995, i64 80
  store float 3.125000e-02, ptr %3015, align 16, !tbaa !5
  %3016 = getelementptr i8, ptr %2995, i64 84
  store float 3.125000e-02, ptr %3016, align 4, !tbaa !5
  %3017 = getelementptr i8, ptr %2995, i64 88
  store float 3.125000e-02, ptr %3017, align 8, !tbaa !5
  %3018 = getelementptr i8, ptr %2995, i64 92
  store float 3.125000e-02, ptr %3018, align 4, !tbaa !5
  %3019 = add nuw nsw i64 %2994, 1
  %exitcond463.not = icmp eq i64 %3019, 4
  br i1 %exitcond463.not, label %.preheader337.preheader, label %.preheader338

.preheader337.preheader:                          ; preds = %.preheader338
  %3020 = load float, ptr %2992, align 4, !tbaa !5
  %3021 = load float, ptr %1326, align 64, !tbaa !5
  %3022 = fmul float %3020, %3021
  store float %3022, ptr %1107, align 64, !tbaa !5
  %3023 = getelementptr inbounds nuw i8, ptr %2992, i64 4
  %3024 = load float, ptr %3023, align 4, !tbaa !5
  %3025 = load float, ptr %1427, align 4, !tbaa !5
  %3026 = fmul float %3024, %3025
  store float %3026, ptr %1215, align 4, !tbaa !5
  %3027 = getelementptr inbounds nuw i8, ptr %2992, i64 8
  %3028 = load float, ptr %3027, align 4, !tbaa !5
  %3029 = load float, ptr %1428, align 8, !tbaa !5
  %3030 = fmul float %3028, %3029
  store float %3030, ptr %1216, align 8, !tbaa !5
  %3031 = getelementptr inbounds nuw i8, ptr %2992, i64 12
  %3032 = load float, ptr %3031, align 4, !tbaa !5
  %3033 = load float, ptr %1429, align 4, !tbaa !5
  %3034 = fmul float %3032, %3033
  store float %3034, ptr %1217, align 4, !tbaa !5
  %3035 = getelementptr inbounds nuw i8, ptr %2992, i64 16
  %3036 = load float, ptr %3035, align 4, !tbaa !5
  %3037 = load float, ptr %1430, align 16, !tbaa !5
  %3038 = fmul float %3036, %3037
  store float %3038, ptr %1218, align 16, !tbaa !5
  %3039 = getelementptr inbounds nuw i8, ptr %2992, i64 20
  %3040 = load float, ptr %3039, align 4, !tbaa !5
  %3041 = load float, ptr %1431, align 4, !tbaa !5
  %3042 = fmul float %3040, %3041
  store float %3042, ptr %1219, align 4, !tbaa !5
  %3043 = getelementptr inbounds nuw i8, ptr %2992, i64 24
  %3044 = load float, ptr %3043, align 4, !tbaa !5
  %3045 = load float, ptr %1432, align 8, !tbaa !5
  %3046 = fmul float %3044, %3045
  store float %3046, ptr %1220, align 8, !tbaa !5
  %3047 = getelementptr inbounds nuw i8, ptr %2992, i64 28
  %3048 = load float, ptr %3047, align 4, !tbaa !5
  %3049 = load float, ptr %1433, align 4, !tbaa !5
  %3050 = fmul float %3048, %3049
  store float %3050, ptr %1221, align 4, !tbaa !5
  %3051 = getelementptr inbounds nuw i8, ptr %2992, i64 32
  %3052 = load float, ptr %3051, align 4, !tbaa !5
  %3053 = load float, ptr %1434, align 32, !tbaa !5
  %3054 = fmul float %3052, %3053
  store float %3054, ptr %1222, align 32, !tbaa !5
  %3055 = getelementptr inbounds nuw i8, ptr %2992, i64 36
  %3056 = load float, ptr %3055, align 4, !tbaa !5
  %3057 = load float, ptr %1435, align 4, !tbaa !5
  %3058 = fmul float %3056, %3057
  store float %3058, ptr %1223, align 4, !tbaa !5
  %3059 = getelementptr inbounds nuw i8, ptr %2992, i64 40
  %3060 = load float, ptr %3059, align 4, !tbaa !5
  %3061 = load float, ptr %1436, align 8, !tbaa !5
  %3062 = fmul float %3060, %3061
  store float %3062, ptr %1224, align 8, !tbaa !5
  %3063 = getelementptr inbounds nuw i8, ptr %2992, i64 44
  %3064 = load float, ptr %3063, align 4, !tbaa !5
  %3065 = load float, ptr %1437, align 4, !tbaa !5
  %3066 = fmul float %3064, %3065
  store float %3066, ptr %1225, align 4, !tbaa !5
  %3067 = getelementptr inbounds nuw i8, ptr %2992, i64 48
  %3068 = load float, ptr %3067, align 4, !tbaa !5
  %3069 = load float, ptr %1438, align 16, !tbaa !5
  %3070 = fmul float %3068, %3069
  store float %3070, ptr %1226, align 16, !tbaa !5
  %3071 = getelementptr inbounds nuw i8, ptr %2992, i64 52
  %3072 = load float, ptr %3071, align 4, !tbaa !5
  %3073 = load float, ptr %1439, align 4, !tbaa !5
  %3074 = fmul float %3072, %3073
  store float %3074, ptr %1227, align 4, !tbaa !5
  %3075 = getelementptr inbounds nuw i8, ptr %2992, i64 56
  %3076 = load float, ptr %3075, align 4, !tbaa !5
  %3077 = load float, ptr %1440, align 8, !tbaa !5
  %3078 = fmul float %3076, %3077
  store float %3078, ptr %1228, align 8, !tbaa !5
  %3079 = getelementptr inbounds nuw i8, ptr %2992, i64 60
  %3080 = load float, ptr %3079, align 4, !tbaa !5
  %3081 = load float, ptr %1441, align 4, !tbaa !5
  %3082 = fmul float %3080, %3081
  store float %3082, ptr %1229, align 4, !tbaa !5
  %3083 = getelementptr inbounds nuw i8, ptr %2992, i64 64
  %3084 = load float, ptr %3083, align 4, !tbaa !5
  %3085 = load float, ptr %1442, align 64, !tbaa !5
  %3086 = fmul float %3084, %3085
  store float %3086, ptr %1230, align 64, !tbaa !5
  %3087 = getelementptr inbounds nuw i8, ptr %2992, i64 68
  %3088 = load float, ptr %3087, align 4, !tbaa !5
  %3089 = load float, ptr %1443, align 4, !tbaa !5
  %3090 = fmul float %3088, %3089
  store float %3090, ptr %1231, align 4, !tbaa !5
  %3091 = getelementptr inbounds nuw i8, ptr %2992, i64 72
  %3092 = load float, ptr %3091, align 4, !tbaa !5
  %3093 = load float, ptr %1444, align 8, !tbaa !5
  %3094 = fmul float %3092, %3093
  store float %3094, ptr %1232, align 8, !tbaa !5
  %3095 = getelementptr inbounds nuw i8, ptr %2992, i64 76
  %3096 = load float, ptr %3095, align 4, !tbaa !5
  %3097 = load float, ptr %1445, align 4, !tbaa !5
  %3098 = fmul float %3096, %3097
  store float %3098, ptr %1233, align 4, !tbaa !5
  %3099 = getelementptr inbounds nuw i8, ptr %2992, i64 80
  %3100 = load float, ptr %3099, align 4, !tbaa !5
  %3101 = load float, ptr %1446, align 16, !tbaa !5
  %3102 = fmul float %3100, %3101
  store float %3102, ptr %1234, align 16, !tbaa !5
  %3103 = getelementptr inbounds nuw i8, ptr %2992, i64 84
  %3104 = load float, ptr %3103, align 4, !tbaa !5
  %3105 = load float, ptr %1447, align 4, !tbaa !5
  %3106 = fmul float %3104, %3105
  store float %3106, ptr %1235, align 4, !tbaa !5
  %3107 = getelementptr inbounds nuw i8, ptr %2992, i64 88
  %3108 = load float, ptr %3107, align 4, !tbaa !5
  %3109 = load float, ptr %1448, align 8, !tbaa !5
  %3110 = fmul float %3108, %3109
  store float %3110, ptr %1236, align 8, !tbaa !5
  %3111 = getelementptr inbounds nuw i8, ptr %2992, i64 92
  %3112 = load float, ptr %3111, align 4, !tbaa !5
  %3113 = load float, ptr %1449, align 4, !tbaa !5
  %3114 = fmul float %3112, %3113
  store float %3114, ptr %1237, align 4, !tbaa !5
  %3115 = getelementptr inbounds nuw i8, ptr %2992, i64 96
  %3116 = load float, ptr %3115, align 4, !tbaa !5
  %3117 = load float, ptr %1450, align 32, !tbaa !5
  %3118 = fmul float %3116, %3117
  store float %3118, ptr %1238, align 32, !tbaa !5
  %3119 = getelementptr inbounds nuw i8, ptr %2992, i64 100
  %3120 = load float, ptr %3119, align 4, !tbaa !5
  %3121 = load float, ptr %1451, align 4, !tbaa !5
  %3122 = fmul float %3120, %3121
  store float %3122, ptr %1239, align 4, !tbaa !5
  %3123 = getelementptr inbounds nuw i8, ptr %2992, i64 104
  %3124 = load float, ptr %3123, align 4, !tbaa !5
  %3125 = load float, ptr %1452, align 8, !tbaa !5
  %3126 = fmul float %3124, %3125
  store float %3126, ptr %1240, align 8, !tbaa !5
  %3127 = getelementptr inbounds nuw i8, ptr %2992, i64 108
  %3128 = load float, ptr %3127, align 4, !tbaa !5
  %3129 = load float, ptr %1453, align 4, !tbaa !5
  %3130 = fmul float %3128, %3129
  store float %3130, ptr %1241, align 4, !tbaa !5
  %3131 = getelementptr inbounds nuw i8, ptr %2992, i64 112
  %3132 = load float, ptr %3131, align 4, !tbaa !5
  %3133 = load float, ptr %1454, align 16, !tbaa !5
  %3134 = fmul float %3132, %3133
  store float %3134, ptr %1242, align 16, !tbaa !5
  %3135 = getelementptr inbounds nuw i8, ptr %2992, i64 116
  %3136 = load float, ptr %3135, align 4, !tbaa !5
  %3137 = load float, ptr %1455, align 4, !tbaa !5
  %3138 = fmul float %3136, %3137
  store float %3138, ptr %1243, align 4, !tbaa !5
  %3139 = getelementptr inbounds nuw i8, ptr %2992, i64 120
  %3140 = load float, ptr %3139, align 4, !tbaa !5
  %3141 = load float, ptr %1456, align 8, !tbaa !5
  %3142 = fmul float %3140, %3141
  store float %3142, ptr %1244, align 8, !tbaa !5
  %3143 = getelementptr inbounds nuw i8, ptr %2992, i64 124
  %3144 = load float, ptr %3143, align 4, !tbaa !5
  %3145 = load float, ptr %1457, align 4, !tbaa !5
  %3146 = fmul float %3144, %3145
  store float %3146, ptr %1245, align 4, !tbaa !5
  %3147 = getelementptr inbounds nuw i8, ptr %2992, i64 128
  %3148 = load float, ptr %3147, align 4, !tbaa !5
  %3149 = load float, ptr %1458, align 64, !tbaa !5
  %3150 = fmul float %3148, %3149
  store float %3150, ptr %1246, align 64, !tbaa !5
  %3151 = getelementptr inbounds nuw i8, ptr %2992, i64 132
  %3152 = load float, ptr %3151, align 4, !tbaa !5
  %3153 = load float, ptr %1459, align 4, !tbaa !5
  %3154 = fmul float %3152, %3153
  store float %3154, ptr %1247, align 4, !tbaa !5
  %3155 = getelementptr inbounds nuw i8, ptr %2992, i64 136
  %3156 = load float, ptr %3155, align 4, !tbaa !5
  %3157 = load float, ptr %1460, align 8, !tbaa !5
  %3158 = fmul float %3156, %3157
  store float %3158, ptr %1248, align 8, !tbaa !5
  %3159 = getelementptr inbounds nuw i8, ptr %2992, i64 140
  %3160 = load float, ptr %3159, align 4, !tbaa !5
  %3161 = load float, ptr %1461, align 4, !tbaa !5
  %3162 = fmul float %3160, %3161
  store float %3162, ptr %1249, align 4, !tbaa !5
  %3163 = getelementptr inbounds nuw i8, ptr %2992, i64 144
  %3164 = load float, ptr %3163, align 4, !tbaa !5
  %3165 = load float, ptr %1462, align 16, !tbaa !5
  %3166 = fmul float %3164, %3165
  store float %3166, ptr %1250, align 16, !tbaa !5
  %3167 = getelementptr inbounds nuw i8, ptr %2992, i64 148
  %3168 = load float, ptr %3167, align 4, !tbaa !5
  %3169 = load float, ptr %1463, align 4, !tbaa !5
  %3170 = fmul float %3168, %3169
  store float %3170, ptr %1251, align 4, !tbaa !5
  %3171 = getelementptr inbounds nuw i8, ptr %2992, i64 152
  %3172 = load float, ptr %3171, align 4, !tbaa !5
  %3173 = load float, ptr %1464, align 8, !tbaa !5
  %3174 = fmul float %3172, %3173
  store float %3174, ptr %1252, align 8, !tbaa !5
  %3175 = getelementptr inbounds nuw i8, ptr %2992, i64 156
  %3176 = load float, ptr %3175, align 4, !tbaa !5
  %3177 = load float, ptr %1465, align 4, !tbaa !5
  %3178 = fmul float %3176, %3177
  store float %3178, ptr %1253, align 4, !tbaa !5
  %3179 = getelementptr inbounds nuw i8, ptr %2992, i64 160
  %3180 = load float, ptr %3179, align 4, !tbaa !5
  %3181 = load float, ptr %1466, align 32, !tbaa !5
  %3182 = fmul float %3180, %3181
  store float %3182, ptr %1254, align 32, !tbaa !5
  %3183 = getelementptr inbounds nuw i8, ptr %2992, i64 164
  %3184 = load float, ptr %3183, align 4, !tbaa !5
  %3185 = load float, ptr %1467, align 4, !tbaa !5
  %3186 = fmul float %3184, %3185
  store float %3186, ptr %1255, align 4, !tbaa !5
  %3187 = getelementptr inbounds nuw i8, ptr %2992, i64 168
  %3188 = load float, ptr %3187, align 4, !tbaa !5
  %3189 = load float, ptr %1468, align 8, !tbaa !5
  %3190 = fmul float %3188, %3189
  store float %3190, ptr %1256, align 8, !tbaa !5
  %3191 = getelementptr inbounds nuw i8, ptr %2992, i64 172
  %3192 = load float, ptr %3191, align 4, !tbaa !5
  %3193 = load float, ptr %1469, align 4, !tbaa !5
  %3194 = fmul float %3192, %3193
  store float %3194, ptr %1257, align 4, !tbaa !5
  %3195 = getelementptr inbounds nuw i8, ptr %2992, i64 176
  %3196 = load float, ptr %3195, align 4, !tbaa !5
  %3197 = load float, ptr %1470, align 16, !tbaa !5
  %3198 = fmul float %3196, %3197
  store float %3198, ptr %1258, align 16, !tbaa !5
  %3199 = getelementptr inbounds nuw i8, ptr %2992, i64 180
  %3200 = load float, ptr %3199, align 4, !tbaa !5
  %3201 = load float, ptr %1471, align 4, !tbaa !5
  %3202 = fmul float %3200, %3201
  store float %3202, ptr %1259, align 4, !tbaa !5
  %3203 = getelementptr inbounds nuw i8, ptr %2992, i64 184
  %3204 = load float, ptr %3203, align 4, !tbaa !5
  %3205 = load float, ptr %1472, align 8, !tbaa !5
  %3206 = fmul float %3204, %3205
  store float %3206, ptr %1260, align 8, !tbaa !5
  %3207 = getelementptr inbounds nuw i8, ptr %2992, i64 188
  %3208 = load float, ptr %3207, align 4, !tbaa !5
  %3209 = load float, ptr %1473, align 4, !tbaa !5
  %3210 = fmul float %3208, %3209
  store float %3210, ptr %1261, align 4, !tbaa !5
  %3211 = getelementptr inbounds nuw i8, ptr %2992, i64 192
  %3212 = load float, ptr %3211, align 4, !tbaa !5
  %3213 = load float, ptr %1474, align 64, !tbaa !5
  %3214 = fmul float %3212, %3213
  store float %3214, ptr %1262, align 64, !tbaa !5
  %3215 = getelementptr inbounds nuw i8, ptr %2992, i64 196
  %3216 = load float, ptr %3215, align 4, !tbaa !5
  %3217 = load float, ptr %1475, align 4, !tbaa !5
  %3218 = fmul float %3216, %3217
  store float %3218, ptr %1263, align 4, !tbaa !5
  %3219 = getelementptr inbounds nuw i8, ptr %2992, i64 200
  %3220 = load float, ptr %3219, align 4, !tbaa !5
  %3221 = load float, ptr %1476, align 8, !tbaa !5
  %3222 = fmul float %3220, %3221
  store float %3222, ptr %1264, align 8, !tbaa !5
  %3223 = getelementptr inbounds nuw i8, ptr %2992, i64 204
  %3224 = load float, ptr %3223, align 4, !tbaa !5
  %3225 = load float, ptr %1477, align 4, !tbaa !5
  %3226 = fmul float %3224, %3225
  store float %3226, ptr %1265, align 4, !tbaa !5
  %3227 = getelementptr inbounds nuw i8, ptr %2992, i64 208
  %3228 = load float, ptr %3227, align 4, !tbaa !5
  %3229 = load float, ptr %1478, align 16, !tbaa !5
  %3230 = fmul float %3228, %3229
  store float %3230, ptr %1266, align 16, !tbaa !5
  %3231 = getelementptr inbounds nuw i8, ptr %2992, i64 212
  %3232 = load float, ptr %3231, align 4, !tbaa !5
  %3233 = load float, ptr %1479, align 4, !tbaa !5
  %3234 = fmul float %3232, %3233
  store float %3234, ptr %1267, align 4, !tbaa !5
  %3235 = getelementptr inbounds nuw i8, ptr %2992, i64 216
  %3236 = load float, ptr %3235, align 4, !tbaa !5
  %3237 = load float, ptr %1480, align 8, !tbaa !5
  %3238 = fmul float %3236, %3237
  store float %3238, ptr %1268, align 8, !tbaa !5
  %3239 = getelementptr inbounds nuw i8, ptr %2992, i64 220
  %3240 = load float, ptr %3239, align 4, !tbaa !5
  %3241 = load float, ptr %1481, align 4, !tbaa !5
  %3242 = fmul float %3240, %3241
  store float %3242, ptr %1269, align 4, !tbaa !5
  %3243 = getelementptr inbounds nuw i8, ptr %2992, i64 224
  %3244 = load float, ptr %3243, align 4, !tbaa !5
  %3245 = load float, ptr %1482, align 32, !tbaa !5
  %3246 = fmul float %3244, %3245
  store float %3246, ptr %1270, align 32, !tbaa !5
  %3247 = getelementptr inbounds nuw i8, ptr %2992, i64 228
  %3248 = load float, ptr %3247, align 4, !tbaa !5
  %3249 = load float, ptr %1483, align 4, !tbaa !5
  %3250 = fmul float %3248, %3249
  store float %3250, ptr %1271, align 4, !tbaa !5
  %3251 = getelementptr inbounds nuw i8, ptr %2992, i64 232
  %3252 = load float, ptr %3251, align 4, !tbaa !5
  %3253 = load float, ptr %1484, align 8, !tbaa !5
  %3254 = fmul float %3252, %3253
  store float %3254, ptr %1272, align 8, !tbaa !5
  %3255 = getelementptr inbounds nuw i8, ptr %2992, i64 236
  %3256 = load float, ptr %3255, align 4, !tbaa !5
  %3257 = load float, ptr %1485, align 4, !tbaa !5
  %3258 = fmul float %3256, %3257
  store float %3258, ptr %1273, align 4, !tbaa !5
  %3259 = getelementptr inbounds nuw i8, ptr %2992, i64 240
  %3260 = load float, ptr %3259, align 4, !tbaa !5
  %3261 = load float, ptr %1486, align 16, !tbaa !5
  %3262 = fmul float %3260, %3261
  store float %3262, ptr %1274, align 16, !tbaa !5
  %3263 = getelementptr inbounds nuw i8, ptr %2992, i64 244
  %3264 = load float, ptr %3263, align 4, !tbaa !5
  %3265 = load float, ptr %1487, align 4, !tbaa !5
  %3266 = fmul float %3264, %3265
  store float %3266, ptr %1275, align 4, !tbaa !5
  %3267 = getelementptr inbounds nuw i8, ptr %2992, i64 248
  %3268 = load float, ptr %3267, align 4, !tbaa !5
  %3269 = load float, ptr %1488, align 8, !tbaa !5
  %3270 = fmul float %3268, %3269
  store float %3270, ptr %1276, align 8, !tbaa !5
  %3271 = getelementptr inbounds nuw i8, ptr %2992, i64 252
  %3272 = load float, ptr %3271, align 4, !tbaa !5
  %3273 = load float, ptr %1489, align 4, !tbaa !5
  %3274 = fmul float %3272, %3273
  store float %3274, ptr %1277, align 4, !tbaa !5
  %3275 = getelementptr inbounds nuw i8, ptr %2992, i64 256
  %3276 = load float, ptr %3275, align 4, !tbaa !5
  %3277 = load float, ptr %1490, align 64, !tbaa !5
  %3278 = fmul float %3276, %3277
  store float %3278, ptr %1278, align 64, !tbaa !5
  %3279 = getelementptr inbounds nuw i8, ptr %2992, i64 260
  %3280 = load float, ptr %3279, align 4, !tbaa !5
  %3281 = load float, ptr %1491, align 4, !tbaa !5
  %3282 = fmul float %3280, %3281
  store float %3282, ptr %1279, align 4, !tbaa !5
  %3283 = getelementptr inbounds nuw i8, ptr %2992, i64 264
  %3284 = load float, ptr %3283, align 4, !tbaa !5
  %3285 = load float, ptr %1492, align 8, !tbaa !5
  %3286 = fmul float %3284, %3285
  store float %3286, ptr %1280, align 8, !tbaa !5
  %3287 = getelementptr inbounds nuw i8, ptr %2992, i64 268
  %3288 = load float, ptr %3287, align 4, !tbaa !5
  %3289 = load float, ptr %1493, align 4, !tbaa !5
  %3290 = fmul float %3288, %3289
  store float %3290, ptr %1281, align 4, !tbaa !5
  %3291 = getelementptr inbounds nuw i8, ptr %2992, i64 272
  %3292 = load float, ptr %3291, align 4, !tbaa !5
  %3293 = load float, ptr %1494, align 16, !tbaa !5
  %3294 = fmul float %3292, %3293
  store float %3294, ptr %1282, align 16, !tbaa !5
  %3295 = getelementptr inbounds nuw i8, ptr %2992, i64 276
  %3296 = load float, ptr %3295, align 4, !tbaa !5
  %3297 = load float, ptr %1495, align 4, !tbaa !5
  %3298 = fmul float %3296, %3297
  store float %3298, ptr %1283, align 4, !tbaa !5
  %3299 = getelementptr inbounds nuw i8, ptr %2992, i64 280
  %3300 = load float, ptr %3299, align 4, !tbaa !5
  %3301 = load float, ptr %1496, align 8, !tbaa !5
  %3302 = fmul float %3300, %3301
  store float %3302, ptr %1284, align 8, !tbaa !5
  %3303 = getelementptr inbounds nuw i8, ptr %2992, i64 284
  %3304 = load float, ptr %3303, align 4, !tbaa !5
  %3305 = load float, ptr %1497, align 4, !tbaa !5
  %3306 = fmul float %3304, %3305
  store float %3306, ptr %1285, align 4, !tbaa !5
  %3307 = getelementptr inbounds nuw i8, ptr %2992, i64 288
  %3308 = load float, ptr %3307, align 4, !tbaa !5
  %3309 = load float, ptr %1498, align 32, !tbaa !5
  %3310 = fmul float %3308, %3309
  store float %3310, ptr %1286, align 32, !tbaa !5
  %3311 = getelementptr inbounds nuw i8, ptr %2992, i64 292
  %3312 = load float, ptr %3311, align 4, !tbaa !5
  %3313 = load float, ptr %1499, align 4, !tbaa !5
  %3314 = fmul float %3312, %3313
  store float %3314, ptr %1287, align 4, !tbaa !5
  %3315 = getelementptr inbounds nuw i8, ptr %2992, i64 296
  %3316 = load float, ptr %3315, align 4, !tbaa !5
  %3317 = load float, ptr %1500, align 8, !tbaa !5
  %3318 = fmul float %3316, %3317
  store float %3318, ptr %1288, align 8, !tbaa !5
  %3319 = getelementptr inbounds nuw i8, ptr %2992, i64 300
  %3320 = load float, ptr %3319, align 4, !tbaa !5
  %3321 = load float, ptr %1501, align 4, !tbaa !5
  %3322 = fmul float %3320, %3321
  store float %3322, ptr %1289, align 4, !tbaa !5
  %3323 = getelementptr inbounds nuw i8, ptr %2992, i64 304
  %3324 = load float, ptr %3323, align 4, !tbaa !5
  %3325 = load float, ptr %1502, align 16, !tbaa !5
  %3326 = fmul float %3324, %3325
  store float %3326, ptr %1290, align 16, !tbaa !5
  %3327 = getelementptr inbounds nuw i8, ptr %2992, i64 308
  %3328 = load float, ptr %3327, align 4, !tbaa !5
  %3329 = load float, ptr %1503, align 4, !tbaa !5
  %3330 = fmul float %3328, %3329
  store float %3330, ptr %1291, align 4, !tbaa !5
  %3331 = getelementptr inbounds nuw i8, ptr %2992, i64 312
  %3332 = load float, ptr %3331, align 4, !tbaa !5
  %3333 = load float, ptr %1504, align 8, !tbaa !5
  %3334 = fmul float %3332, %3333
  store float %3334, ptr %1292, align 8, !tbaa !5
  %3335 = getelementptr inbounds nuw i8, ptr %2992, i64 316
  %3336 = load float, ptr %3335, align 4, !tbaa !5
  %3337 = load float, ptr %1505, align 4, !tbaa !5
  %3338 = fmul float %3336, %3337
  store float %3338, ptr %1293, align 4, !tbaa !5
  %3339 = getelementptr inbounds nuw i8, ptr %2992, i64 320
  %3340 = load float, ptr %3339, align 4, !tbaa !5
  %3341 = load float, ptr %1506, align 64, !tbaa !5
  %3342 = fmul float %3340, %3341
  store float %3342, ptr %1294, align 64, !tbaa !5
  %3343 = getelementptr inbounds nuw i8, ptr %2992, i64 324
  %3344 = load float, ptr %3343, align 4, !tbaa !5
  %3345 = load float, ptr %1507, align 4, !tbaa !5
  %3346 = fmul float %3344, %3345
  store float %3346, ptr %1295, align 4, !tbaa !5
  %3347 = getelementptr inbounds nuw i8, ptr %2992, i64 328
  %3348 = load float, ptr %3347, align 4, !tbaa !5
  %3349 = load float, ptr %1508, align 8, !tbaa !5
  %3350 = fmul float %3348, %3349
  store float %3350, ptr %1296, align 8, !tbaa !5
  %3351 = getelementptr inbounds nuw i8, ptr %2992, i64 332
  %3352 = load float, ptr %3351, align 4, !tbaa !5
  %3353 = load float, ptr %1509, align 4, !tbaa !5
  %3354 = fmul float %3352, %3353
  store float %3354, ptr %1297, align 4, !tbaa !5
  %3355 = getelementptr inbounds nuw i8, ptr %2992, i64 336
  %3356 = load float, ptr %3355, align 4, !tbaa !5
  %3357 = load float, ptr %1510, align 16, !tbaa !5
  %3358 = fmul float %3356, %3357
  store float %3358, ptr %1298, align 16, !tbaa !5
  %3359 = getelementptr inbounds nuw i8, ptr %2992, i64 340
  %3360 = load float, ptr %3359, align 4, !tbaa !5
  %3361 = load float, ptr %1511, align 4, !tbaa !5
  %3362 = fmul float %3360, %3361
  store float %3362, ptr %1299, align 4, !tbaa !5
  %3363 = getelementptr inbounds nuw i8, ptr %2992, i64 344
  %3364 = load float, ptr %3363, align 4, !tbaa !5
  %3365 = load float, ptr %1512, align 8, !tbaa !5
  %3366 = fmul float %3364, %3365
  store float %3366, ptr %1300, align 8, !tbaa !5
  %3367 = getelementptr inbounds nuw i8, ptr %2992, i64 348
  %3368 = load float, ptr %3367, align 4, !tbaa !5
  %3369 = load float, ptr %1513, align 4, !tbaa !5
  %3370 = fmul float %3368, %3369
  store float %3370, ptr %1301, align 4, !tbaa !5
  %3371 = getelementptr inbounds nuw i8, ptr %2992, i64 352
  %3372 = load float, ptr %3371, align 4, !tbaa !5
  %3373 = load float, ptr %1514, align 32, !tbaa !5
  %3374 = fmul float %3372, %3373
  store float %3374, ptr %1302, align 32, !tbaa !5
  %3375 = getelementptr inbounds nuw i8, ptr %2992, i64 356
  %3376 = load float, ptr %3375, align 4, !tbaa !5
  %3377 = load float, ptr %1515, align 4, !tbaa !5
  %3378 = fmul float %3376, %3377
  store float %3378, ptr %1303, align 4, !tbaa !5
  %3379 = getelementptr inbounds nuw i8, ptr %2992, i64 360
  %3380 = load float, ptr %3379, align 4, !tbaa !5
  %3381 = load float, ptr %1516, align 8, !tbaa !5
  %3382 = fmul float %3380, %3381
  store float %3382, ptr %1304, align 8, !tbaa !5
  %3383 = getelementptr inbounds nuw i8, ptr %2992, i64 364
  %3384 = load float, ptr %3383, align 4, !tbaa !5
  %3385 = load float, ptr %1517, align 4, !tbaa !5
  %3386 = fmul float %3384, %3385
  store float %3386, ptr %1305, align 4, !tbaa !5
  %3387 = getelementptr inbounds nuw i8, ptr %2992, i64 368
  %3388 = load float, ptr %3387, align 4, !tbaa !5
  %3389 = load float, ptr %1518, align 16, !tbaa !5
  %3390 = fmul float %3388, %3389
  store float %3390, ptr %1306, align 16, !tbaa !5
  %3391 = getelementptr inbounds nuw i8, ptr %2992, i64 372
  %3392 = load float, ptr %3391, align 4, !tbaa !5
  %3393 = load float, ptr %1519, align 4, !tbaa !5
  %3394 = fmul float %3392, %3393
  store float %3394, ptr %1307, align 4, !tbaa !5
  %3395 = getelementptr inbounds nuw i8, ptr %2992, i64 376
  %3396 = load float, ptr %3395, align 4, !tbaa !5
  %3397 = load float, ptr %1520, align 8, !tbaa !5
  %3398 = fmul float %3396, %3397
  store float %3398, ptr %1308, align 8, !tbaa !5
  %3399 = getelementptr inbounds nuw i8, ptr %2992, i64 380
  %3400 = load float, ptr %3399, align 4, !tbaa !5
  %3401 = load float, ptr %1521, align 4, !tbaa !5
  %3402 = fmul float %3400, %3401
  store float %3402, ptr %1309, align 4, !tbaa !5
  tail call void @_mlir_memref_to_llvm_free(ptr nonnull %2992)
  br label %.preheader336

.preheader336:                                    ; preds = %.preheader337.preheader, %.preheader336
  %3403 = phi i64 [ 0, %.preheader337.preheader ], [ %3428, %.preheader336 ]
  %.idx = mul nuw nsw i64 %3403, 96
  %3404 = getelementptr i8, ptr %1326, i64 %.idx
  store float 0x3F847AE140000000, ptr %3404, align 32, !tbaa !5
  %3405 = getelementptr i8, ptr %3404, i64 4
  store float 0x3F847AE140000000, ptr %3405, align 4, !tbaa !5
  %3406 = getelementptr i8, ptr %3404, i64 8
  store float 0x3F847AE140000000, ptr %3406, align 8, !tbaa !5
  %3407 = getelementptr i8, ptr %3404, i64 12
  store float 0x3F847AE140000000, ptr %3407, align 4, !tbaa !5
  %3408 = getelementptr i8, ptr %3404, i64 16
  store float 0x3F847AE140000000, ptr %3408, align 16, !tbaa !5
  %3409 = getelementptr i8, ptr %3404, i64 20
  store float 0x3F847AE140000000, ptr %3409, align 4, !tbaa !5
  %3410 = getelementptr i8, ptr %3404, i64 24
  store float 0x3F847AE140000000, ptr %3410, align 8, !tbaa !5
  %3411 = getelementptr i8, ptr %3404, i64 28
  store float 0x3F847AE140000000, ptr %3411, align 4, !tbaa !5
  %3412 = getelementptr i8, ptr %3404, i64 32
  store float 0x3F847AE140000000, ptr %3412, align 32, !tbaa !5
  %3413 = getelementptr i8, ptr %3404, i64 36
  store float 0x3F847AE140000000, ptr %3413, align 4, !tbaa !5
  %3414 = getelementptr i8, ptr %3404, i64 40
  store float 0x3F847AE140000000, ptr %3414, align 8, !tbaa !5
  %3415 = getelementptr i8, ptr %3404, i64 44
  store float 0x3F847AE140000000, ptr %3415, align 4, !tbaa !5
  %3416 = getelementptr i8, ptr %3404, i64 48
  store float 0x3F847AE140000000, ptr %3416, align 16, !tbaa !5
  %3417 = getelementptr i8, ptr %3404, i64 52
  store float 0x3F847AE140000000, ptr %3417, align 4, !tbaa !5
  %3418 = getelementptr i8, ptr %3404, i64 56
  store float 0x3F847AE140000000, ptr %3418, align 8, !tbaa !5
  %3419 = getelementptr i8, ptr %3404, i64 60
  store float 0x3F847AE140000000, ptr %3419, align 4, !tbaa !5
  %3420 = getelementptr i8, ptr %3404, i64 64
  store float 0x3F847AE140000000, ptr %3420, align 32, !tbaa !5
  %3421 = getelementptr i8, ptr %3404, i64 68
  store float 0x3F847AE140000000, ptr %3421, align 4, !tbaa !5
  %3422 = getelementptr i8, ptr %3404, i64 72
  store float 0x3F847AE140000000, ptr %3422, align 8, !tbaa !5
  %3423 = getelementptr i8, ptr %3404, i64 76
  store float 0x3F847AE140000000, ptr %3423, align 4, !tbaa !5
  %3424 = getelementptr i8, ptr %3404, i64 80
  store float 0x3F847AE140000000, ptr %3424, align 16, !tbaa !5
  %3425 = getelementptr i8, ptr %3404, i64 84
  store float 0x3F847AE140000000, ptr %3425, align 4, !tbaa !5
  %3426 = getelementptr i8, ptr %3404, i64 88
  store float 0x3F847AE140000000, ptr %3426, align 8, !tbaa !5
  %3427 = getelementptr i8, ptr %3404, i64 92
  store float 0x3F847AE140000000, ptr %3427, align 4, !tbaa !5
  %3428 = add nuw nsw i64 %3403, 1
  %exitcond472.not = icmp eq i64 %3428, 4
  br i1 %exitcond472.not, label %.preheader335, label %.preheader336

.preheader335:                                    ; preds = %.preheader336, %.preheader335
  %3429 = phi i64 [ %3574, %.preheader335 ], [ 0, %.preheader336 ]
  %3430 = mul nuw nsw i64 %3429, 24
  %3431 = getelementptr inbounds nuw float, ptr %1326, i64 %3430
  %3432 = load float, ptr %3431, align 32, !tbaa !5
  %3433 = getelementptr inbounds nuw float, ptr %1107, i64 %3430
  %3434 = load float, ptr %3433, align 32, !tbaa !5
  %3435 = fmul float %3432, %3434
  store float %3435, ptr %3431, align 32, !tbaa !5
  %3436 = or disjoint i64 %3430, 1
  %3437 = getelementptr inbounds nuw float, ptr %1326, i64 %3436
  %3438 = load float, ptr %3437, align 4, !tbaa !5
  %3439 = getelementptr inbounds nuw float, ptr %1107, i64 %3436
  %3440 = load float, ptr %3439, align 4, !tbaa !5
  %3441 = fmul float %3438, %3440
  store float %3441, ptr %3437, align 4, !tbaa !5
  %3442 = or disjoint i64 %3430, 2
  %3443 = getelementptr inbounds nuw float, ptr %1326, i64 %3442
  %3444 = load float, ptr %3443, align 8, !tbaa !5
  %3445 = getelementptr inbounds nuw float, ptr %1107, i64 %3442
  %3446 = load float, ptr %3445, align 8, !tbaa !5
  %3447 = fmul float %3444, %3446
  store float %3447, ptr %3443, align 8, !tbaa !5
  %3448 = or disjoint i64 %3430, 3
  %3449 = getelementptr inbounds nuw float, ptr %1326, i64 %3448
  %3450 = load float, ptr %3449, align 4, !tbaa !5
  %3451 = getelementptr inbounds nuw float, ptr %1107, i64 %3448
  %3452 = load float, ptr %3451, align 4, !tbaa !5
  %3453 = fmul float %3450, %3452
  store float %3453, ptr %3449, align 4, !tbaa !5
  %3454 = or disjoint i64 %3430, 4
  %3455 = getelementptr inbounds nuw float, ptr %1326, i64 %3454
  %3456 = load float, ptr %3455, align 16, !tbaa !5
  %3457 = getelementptr inbounds nuw float, ptr %1107, i64 %3454
  %3458 = load float, ptr %3457, align 16, !tbaa !5
  %3459 = fmul float %3456, %3458
  store float %3459, ptr %3455, align 16, !tbaa !5
  %3460 = or disjoint i64 %3430, 5
  %3461 = getelementptr inbounds nuw float, ptr %1326, i64 %3460
  %3462 = load float, ptr %3461, align 4, !tbaa !5
  %3463 = getelementptr inbounds nuw float, ptr %1107, i64 %3460
  %3464 = load float, ptr %3463, align 4, !tbaa !5
  %3465 = fmul float %3462, %3464
  store float %3465, ptr %3461, align 4, !tbaa !5
  %3466 = or disjoint i64 %3430, 6
  %3467 = getelementptr inbounds nuw float, ptr %1326, i64 %3466
  %3468 = load float, ptr %3467, align 8, !tbaa !5
  %3469 = getelementptr inbounds nuw float, ptr %1107, i64 %3466
  %3470 = load float, ptr %3469, align 8, !tbaa !5
  %3471 = fmul float %3468, %3470
  store float %3471, ptr %3467, align 8, !tbaa !5
  %3472 = or disjoint i64 %3430, 7
  %3473 = getelementptr inbounds nuw float, ptr %1326, i64 %3472
  %3474 = load float, ptr %3473, align 4, !tbaa !5
  %3475 = getelementptr inbounds nuw float, ptr %1107, i64 %3472
  %3476 = load float, ptr %3475, align 4, !tbaa !5
  %3477 = fmul float %3474, %3476
  store float %3477, ptr %3473, align 4, !tbaa !5
  %3478 = add nuw nsw i64 %3430, 8
  %3479 = getelementptr inbounds nuw float, ptr %1326, i64 %3478
  %3480 = load float, ptr %3479, align 32, !tbaa !5
  %3481 = getelementptr inbounds nuw float, ptr %1107, i64 %3478
  %3482 = load float, ptr %3481, align 32, !tbaa !5
  %3483 = fmul float %3480, %3482
  store float %3483, ptr %3479, align 32, !tbaa !5
  %3484 = add nuw nsw i64 %3430, 9
  %3485 = getelementptr inbounds nuw float, ptr %1326, i64 %3484
  %3486 = load float, ptr %3485, align 4, !tbaa !5
  %3487 = getelementptr inbounds nuw float, ptr %1107, i64 %3484
  %3488 = load float, ptr %3487, align 4, !tbaa !5
  %3489 = fmul float %3486, %3488
  store float %3489, ptr %3485, align 4, !tbaa !5
  %3490 = add nuw nsw i64 %3430, 10
  %3491 = getelementptr inbounds nuw float, ptr %1326, i64 %3490
  %3492 = load float, ptr %3491, align 8, !tbaa !5
  %3493 = getelementptr inbounds nuw float, ptr %1107, i64 %3490
  %3494 = load float, ptr %3493, align 8, !tbaa !5
  %3495 = fmul float %3492, %3494
  store float %3495, ptr %3491, align 8, !tbaa !5
  %3496 = add nuw nsw i64 %3430, 11
  %3497 = getelementptr inbounds nuw float, ptr %1326, i64 %3496
  %3498 = load float, ptr %3497, align 4, !tbaa !5
  %3499 = getelementptr inbounds nuw float, ptr %1107, i64 %3496
  %3500 = load float, ptr %3499, align 4, !tbaa !5
  %3501 = fmul float %3498, %3500
  store float %3501, ptr %3497, align 4, !tbaa !5
  %3502 = add nuw nsw i64 %3430, 12
  %3503 = getelementptr inbounds nuw float, ptr %1326, i64 %3502
  %3504 = load float, ptr %3503, align 16, !tbaa !5
  %3505 = getelementptr inbounds nuw float, ptr %1107, i64 %3502
  %3506 = load float, ptr %3505, align 16, !tbaa !5
  %3507 = fmul float %3504, %3506
  store float %3507, ptr %3503, align 16, !tbaa !5
  %3508 = add nuw nsw i64 %3430, 13
  %3509 = getelementptr inbounds nuw float, ptr %1326, i64 %3508
  %3510 = load float, ptr %3509, align 4, !tbaa !5
  %3511 = getelementptr inbounds nuw float, ptr %1107, i64 %3508
  %3512 = load float, ptr %3511, align 4, !tbaa !5
  %3513 = fmul float %3510, %3512
  store float %3513, ptr %3509, align 4, !tbaa !5
  %3514 = add nuw nsw i64 %3430, 14
  %3515 = getelementptr inbounds nuw float, ptr %1326, i64 %3514
  %3516 = load float, ptr %3515, align 8, !tbaa !5
  %3517 = getelementptr inbounds nuw float, ptr %1107, i64 %3514
  %3518 = load float, ptr %3517, align 8, !tbaa !5
  %3519 = fmul float %3516, %3518
  store float %3519, ptr %3515, align 8, !tbaa !5
  %3520 = add nuw nsw i64 %3430, 15
  %3521 = getelementptr inbounds nuw float, ptr %1326, i64 %3520
  %3522 = load float, ptr %3521, align 4, !tbaa !5
  %3523 = getelementptr inbounds nuw float, ptr %1107, i64 %3520
  %3524 = load float, ptr %3523, align 4, !tbaa !5
  %3525 = fmul float %3522, %3524
  store float %3525, ptr %3521, align 4, !tbaa !5
  %3526 = add nuw nsw i64 %3430, 16
  %3527 = getelementptr inbounds nuw float, ptr %1326, i64 %3526
  %3528 = load float, ptr %3527, align 32, !tbaa !5
  %3529 = getelementptr inbounds nuw float, ptr %1107, i64 %3526
  %3530 = load float, ptr %3529, align 32, !tbaa !5
  %3531 = fmul float %3528, %3530
  store float %3531, ptr %3527, align 32, !tbaa !5
  %3532 = add nuw nsw i64 %3430, 17
  %3533 = getelementptr inbounds nuw float, ptr %1326, i64 %3532
  %3534 = load float, ptr %3533, align 4, !tbaa !5
  %3535 = getelementptr inbounds nuw float, ptr %1107, i64 %3532
  %3536 = load float, ptr %3535, align 4, !tbaa !5
  %3537 = fmul float %3534, %3536
  store float %3537, ptr %3533, align 4, !tbaa !5
  %3538 = add nuw nsw i64 %3430, 18
  %3539 = getelementptr inbounds nuw float, ptr %1326, i64 %3538
  %3540 = load float, ptr %3539, align 8, !tbaa !5
  %3541 = getelementptr inbounds nuw float, ptr %1107, i64 %3538
  %3542 = load float, ptr %3541, align 8, !tbaa !5
  %3543 = fmul float %3540, %3542
  store float %3543, ptr %3539, align 8, !tbaa !5
  %3544 = add nuw nsw i64 %3430, 19
  %3545 = getelementptr inbounds nuw float, ptr %1326, i64 %3544
  %3546 = load float, ptr %3545, align 4, !tbaa !5
  %3547 = getelementptr inbounds nuw float, ptr %1107, i64 %3544
  %3548 = load float, ptr %3547, align 4, !tbaa !5
  %3549 = fmul float %3546, %3548
  store float %3549, ptr %3545, align 4, !tbaa !5
  %3550 = add nuw nsw i64 %3430, 20
  %3551 = getelementptr inbounds nuw float, ptr %1326, i64 %3550
  %3552 = load float, ptr %3551, align 16, !tbaa !5
  %3553 = getelementptr inbounds nuw float, ptr %1107, i64 %3550
  %3554 = load float, ptr %3553, align 16, !tbaa !5
  %3555 = fmul float %3552, %3554
  store float %3555, ptr %3551, align 16, !tbaa !5
  %3556 = add nuw nsw i64 %3430, 21
  %3557 = getelementptr inbounds nuw float, ptr %1326, i64 %3556
  %3558 = load float, ptr %3557, align 4, !tbaa !5
  %3559 = getelementptr inbounds nuw float, ptr %1107, i64 %3556
  %3560 = load float, ptr %3559, align 4, !tbaa !5
  %3561 = fmul float %3558, %3560
  store float %3561, ptr %3557, align 4, !tbaa !5
  %3562 = add nuw nsw i64 %3430, 22
  %3563 = getelementptr inbounds nuw float, ptr %1326, i64 %3562
  %3564 = load float, ptr %3563, align 8, !tbaa !5
  %3565 = getelementptr inbounds nuw float, ptr %1107, i64 %3562
  %3566 = load float, ptr %3565, align 8, !tbaa !5
  %3567 = fmul float %3564, %3566
  store float %3567, ptr %3563, align 8, !tbaa !5
  %3568 = add nuw nsw i64 %3430, 23
  %3569 = getelementptr inbounds nuw float, ptr %1326, i64 %3568
  %3570 = load float, ptr %3569, align 4, !tbaa !5
  %3571 = getelementptr inbounds nuw float, ptr %1107, i64 %3568
  %3572 = load float, ptr %3571, align 4, !tbaa !5
  %3573 = fmul float %3570, %3572
  store float %3573, ptr %3569, align 4, !tbaa !5
  %3574 = add nuw nsw i64 %3429, 1
  %exitcond475.not = icmp eq i64 %3574, 4
  br i1 %exitcond475.not, label %.preheader334, label %.preheader335

.preheader334:                                    ; preds = %.preheader335, %.preheader334
  %3575 = phi i64 [ %3720, %.preheader334 ], [ 0, %.preheader335 ]
  %3576 = mul nuw nsw i64 %3575, 24
  %3577 = getelementptr inbounds nuw float, ptr %.pn241419, i64 %3576
  %3578 = load float, ptr %3577, align 4, !tbaa !5
  %3579 = getelementptr inbounds nuw float, ptr %1326, i64 %3576
  %3580 = load float, ptr %3579, align 32, !tbaa !5
  %3581 = fsub float %3578, %3580
  store float %3581, ptr %3579, align 32, !tbaa !5
  %3582 = or disjoint i64 %3576, 1
  %3583 = getelementptr inbounds nuw float, ptr %.pn241419, i64 %3582
  %3584 = load float, ptr %3583, align 4, !tbaa !5
  %3585 = getelementptr inbounds nuw float, ptr %1326, i64 %3582
  %3586 = load float, ptr %3585, align 4, !tbaa !5
  %3587 = fsub float %3584, %3586
  store float %3587, ptr %3585, align 4, !tbaa !5
  %3588 = or disjoint i64 %3576, 2
  %3589 = getelementptr inbounds nuw float, ptr %.pn241419, i64 %3588
  %3590 = load float, ptr %3589, align 4, !tbaa !5
  %3591 = getelementptr inbounds nuw float, ptr %1326, i64 %3588
  %3592 = load float, ptr %3591, align 8, !tbaa !5
  %3593 = fsub float %3590, %3592
  store float %3593, ptr %3591, align 8, !tbaa !5
  %3594 = or disjoint i64 %3576, 3
  %3595 = getelementptr inbounds nuw float, ptr %.pn241419, i64 %3594
  %3596 = load float, ptr %3595, align 4, !tbaa !5
  %3597 = getelementptr inbounds nuw float, ptr %1326, i64 %3594
  %3598 = load float, ptr %3597, align 4, !tbaa !5
  %3599 = fsub float %3596, %3598
  store float %3599, ptr %3597, align 4, !tbaa !5
  %3600 = or disjoint i64 %3576, 4
  %3601 = getelementptr inbounds nuw float, ptr %.pn241419, i64 %3600
  %3602 = load float, ptr %3601, align 4, !tbaa !5
  %3603 = getelementptr inbounds nuw float, ptr %1326, i64 %3600
  %3604 = load float, ptr %3603, align 16, !tbaa !5
  %3605 = fsub float %3602, %3604
  store float %3605, ptr %3603, align 16, !tbaa !5
  %3606 = or disjoint i64 %3576, 5
  %3607 = getelementptr inbounds nuw float, ptr %.pn241419, i64 %3606
  %3608 = load float, ptr %3607, align 4, !tbaa !5
  %3609 = getelementptr inbounds nuw float, ptr %1326, i64 %3606
  %3610 = load float, ptr %3609, align 4, !tbaa !5
  %3611 = fsub float %3608, %3610
  store float %3611, ptr %3609, align 4, !tbaa !5
  %3612 = or disjoint i64 %3576, 6
  %3613 = getelementptr inbounds nuw float, ptr %.pn241419, i64 %3612
  %3614 = load float, ptr %3613, align 4, !tbaa !5
  %3615 = getelementptr inbounds nuw float, ptr %1326, i64 %3612
  %3616 = load float, ptr %3615, align 8, !tbaa !5
  %3617 = fsub float %3614, %3616
  store float %3617, ptr %3615, align 8, !tbaa !5
  %3618 = or disjoint i64 %3576, 7
  %3619 = getelementptr inbounds nuw float, ptr %.pn241419, i64 %3618
  %3620 = load float, ptr %3619, align 4, !tbaa !5
  %3621 = getelementptr inbounds nuw float, ptr %1326, i64 %3618
  %3622 = load float, ptr %3621, align 4, !tbaa !5
  %3623 = fsub float %3620, %3622
  store float %3623, ptr %3621, align 4, !tbaa !5
  %3624 = add nuw nsw i64 %3576, 8
  %3625 = getelementptr inbounds nuw float, ptr %.pn241419, i64 %3624
  %3626 = load float, ptr %3625, align 4, !tbaa !5
  %3627 = getelementptr inbounds nuw float, ptr %1326, i64 %3624
  %3628 = load float, ptr %3627, align 32, !tbaa !5
  %3629 = fsub float %3626, %3628
  store float %3629, ptr %3627, align 32, !tbaa !5
  %3630 = add nuw nsw i64 %3576, 9
  %3631 = getelementptr inbounds nuw float, ptr %.pn241419, i64 %3630
  %3632 = load float, ptr %3631, align 4, !tbaa !5
  %3633 = getelementptr inbounds nuw float, ptr %1326, i64 %3630
  %3634 = load float, ptr %3633, align 4, !tbaa !5
  %3635 = fsub float %3632, %3634
  store float %3635, ptr %3633, align 4, !tbaa !5
  %3636 = add nuw nsw i64 %3576, 10
  %3637 = getelementptr inbounds nuw float, ptr %.pn241419, i64 %3636
  %3638 = load float, ptr %3637, align 4, !tbaa !5
  %3639 = getelementptr inbounds nuw float, ptr %1326, i64 %3636
  %3640 = load float, ptr %3639, align 8, !tbaa !5
  %3641 = fsub float %3638, %3640
  store float %3641, ptr %3639, align 8, !tbaa !5
  %3642 = add nuw nsw i64 %3576, 11
  %3643 = getelementptr inbounds nuw float, ptr %.pn241419, i64 %3642
  %3644 = load float, ptr %3643, align 4, !tbaa !5
  %3645 = getelementptr inbounds nuw float, ptr %1326, i64 %3642
  %3646 = load float, ptr %3645, align 4, !tbaa !5
  %3647 = fsub float %3644, %3646
  store float %3647, ptr %3645, align 4, !tbaa !5
  %3648 = add nuw nsw i64 %3576, 12
  %3649 = getelementptr inbounds nuw float, ptr %.pn241419, i64 %3648
  %3650 = load float, ptr %3649, align 4, !tbaa !5
  %3651 = getelementptr inbounds nuw float, ptr %1326, i64 %3648
  %3652 = load float, ptr %3651, align 16, !tbaa !5
  %3653 = fsub float %3650, %3652
  store float %3653, ptr %3651, align 16, !tbaa !5
  %3654 = add nuw nsw i64 %3576, 13
  %3655 = getelementptr inbounds nuw float, ptr %.pn241419, i64 %3654
  %3656 = load float, ptr %3655, align 4, !tbaa !5
  %3657 = getelementptr inbounds nuw float, ptr %1326, i64 %3654
  %3658 = load float, ptr %3657, align 4, !tbaa !5
  %3659 = fsub float %3656, %3658
  store float %3659, ptr %3657, align 4, !tbaa !5
  %3660 = add nuw nsw i64 %3576, 14
  %3661 = getelementptr inbounds nuw float, ptr %.pn241419, i64 %3660
  %3662 = load float, ptr %3661, align 4, !tbaa !5
  %3663 = getelementptr inbounds nuw float, ptr %1326, i64 %3660
  %3664 = load float, ptr %3663, align 8, !tbaa !5
  %3665 = fsub float %3662, %3664
  store float %3665, ptr %3663, align 8, !tbaa !5
  %3666 = add nuw nsw i64 %3576, 15
  %3667 = getelementptr inbounds nuw float, ptr %.pn241419, i64 %3666
  %3668 = load float, ptr %3667, align 4, !tbaa !5
  %3669 = getelementptr inbounds nuw float, ptr %1326, i64 %3666
  %3670 = load float, ptr %3669, align 4, !tbaa !5
  %3671 = fsub float %3668, %3670
  store float %3671, ptr %3669, align 4, !tbaa !5
  %3672 = add nuw nsw i64 %3576, 16
  %3673 = getelementptr inbounds nuw float, ptr %.pn241419, i64 %3672
  %3674 = load float, ptr %3673, align 4, !tbaa !5
  %3675 = getelementptr inbounds nuw float, ptr %1326, i64 %3672
  %3676 = load float, ptr %3675, align 32, !tbaa !5
  %3677 = fsub float %3674, %3676
  store float %3677, ptr %3675, align 32, !tbaa !5
  %3678 = add nuw nsw i64 %3576, 17
  %3679 = getelementptr inbounds nuw float, ptr %.pn241419, i64 %3678
  %3680 = load float, ptr %3679, align 4, !tbaa !5
  %3681 = getelementptr inbounds nuw float, ptr %1326, i64 %3678
  %3682 = load float, ptr %3681, align 4, !tbaa !5
  %3683 = fsub float %3680, %3682
  store float %3683, ptr %3681, align 4, !tbaa !5
  %3684 = add nuw nsw i64 %3576, 18
  %3685 = getelementptr inbounds nuw float, ptr %.pn241419, i64 %3684
  %3686 = load float, ptr %3685, align 4, !tbaa !5
  %3687 = getelementptr inbounds nuw float, ptr %1326, i64 %3684
  %3688 = load float, ptr %3687, align 8, !tbaa !5
  %3689 = fsub float %3686, %3688
  store float %3689, ptr %3687, align 8, !tbaa !5
  %3690 = add nuw nsw i64 %3576, 19
  %3691 = getelementptr inbounds nuw float, ptr %.pn241419, i64 %3690
  %3692 = load float, ptr %3691, align 4, !tbaa !5
  %3693 = getelementptr inbounds nuw float, ptr %1326, i64 %3690
  %3694 = load float, ptr %3693, align 4, !tbaa !5
  %3695 = fsub float %3692, %3694
  store float %3695, ptr %3693, align 4, !tbaa !5
  %3696 = add nuw nsw i64 %3576, 20
  %3697 = getelementptr inbounds nuw float, ptr %.pn241419, i64 %3696
  %3698 = load float, ptr %3697, align 4, !tbaa !5
  %3699 = getelementptr inbounds nuw float, ptr %1326, i64 %3696
  %3700 = load float, ptr %3699, align 16, !tbaa !5
  %3701 = fsub float %3698, %3700
  store float %3701, ptr %3699, align 16, !tbaa !5
  %3702 = add nuw nsw i64 %3576, 21
  %3703 = getelementptr inbounds nuw float, ptr %.pn241419, i64 %3702
  %3704 = load float, ptr %3703, align 4, !tbaa !5
  %3705 = getelementptr inbounds nuw float, ptr %1326, i64 %3702
  %3706 = load float, ptr %3705, align 4, !tbaa !5
  %3707 = fsub float %3704, %3706
  store float %3707, ptr %3705, align 4, !tbaa !5
  %3708 = add nuw nsw i64 %3576, 22
  %3709 = getelementptr inbounds nuw float, ptr %.pn241419, i64 %3708
  %3710 = load float, ptr %3709, align 4, !tbaa !5
  %3711 = getelementptr inbounds nuw float, ptr %1326, i64 %3708
  %3712 = load float, ptr %3711, align 8, !tbaa !5
  %3713 = fsub float %3710, %3712
  store float %3713, ptr %3711, align 8, !tbaa !5
  %3714 = add nuw nsw i64 %3576, 23
  %3715 = getelementptr inbounds nuw float, ptr %.pn241419, i64 %3714
  %3716 = load float, ptr %3715, align 4, !tbaa !5
  %3717 = getelementptr inbounds nuw float, ptr %1326, i64 %3714
  %3718 = load float, ptr %3717, align 4, !tbaa !5
  %3719 = fsub float %3716, %3718
  store float %3719, ptr %3717, align 4, !tbaa !5
  %3720 = add nuw nsw i64 %3575, 1
  %exitcond478.not = icmp eq i64 %3720, 4
  br i1 %exitcond478.not, label %3721, label %.preheader334

3721:                                             ; preds = %.preheader334
  tail call void @_mlir_memref_to_llvm_free(ptr nonnull %.pn241419)
  %3722 = fmul double %2983, 3.125000e-02
  %3723 = fptrunc double %3722 to float
  %3724 = fmul float %3723, 0x3F847AE140000000
  %3725 = fsub float %1320, %3724
  %3726 = fmul double %2985, 3.125000e-02
  %3727 = fptrunc double %3726 to float
  %3728 = fmul float %3727, 0x3F847AE140000000
  %3729 = fsub float %1319, %3728
  %3730 = fmul double %2986, 3.125000e-02
  %3731 = fadd double %1318, %3730
  %3732 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 384)
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(384) %3732, ptr noundef nonnull align 64 dereferenceable(384) %1326, i64 384, i1 false)
  tail call void @_mlir_memref_to_llvm_free(ptr %1322)
  %3733 = add nuw nsw i64 %1321, 1
  %exitcond479.not = icmp eq i64 %3733, 3
  br i1 %exitcond479.not, label %3734, label %1317

3734:                                             ; preds = %3721
  tail call void @_mlir_memref_to_llvm_free(ptr %1103)
  tail call void @_mlir_memref_to_llvm_free(ptr %1102)
  tail call void @_mlir_memref_to_llvm_free(ptr nonnull %1101)
  tail call void @_mlir_memref_to_llvm_free(ptr nonnull %1100)
  tail call void @_mlir_memref_to_llvm_free(ptr nonnull %1099)
  tail call void @_mlir_memref_to_llvm_free(ptr %1094)
  tail call void @_mlir_memref_to_llvm_free(ptr %1089)
  tail call void @_mlir_memref_to_llvm_free(ptr %1084)
  tail call void @_mlir_memref_to_llvm_free(ptr %1079)
  tail call void @_mlir_memref_to_llvm_free(ptr %1074)
  %3735 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 68)
  %3736 = ptrtoint ptr %3735 to i64
  %3737 = add i64 %3736, 63
  %3738 = and i64 %3737, -64
  %3739 = inttoptr i64 %3738 to ptr
  store float %3725, ptr %3739, align 64, !tbaa !5
  %3740 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 68)
  %3741 = ptrtoint ptr %3740 to i64
  %3742 = add i64 %3741, 63
  %3743 = and i64 %3742, -64
  %3744 = inttoptr i64 %3743 to ptr
  store float %3729, ptr %3744, align 64, !tbaa !5
  %3745 = fdiv double %3731, 3.000000e+00
  %3746 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %3747 = ptrtoint ptr %3746 to i64
  %3748 = add i64 %3747, 63
  %3749 = and i64 %3748, -64
  %3750 = inttoptr i64 %3749 to ptr
  store double %3745, ptr %3750, align 64, !tbaa !7
  %3751 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %3752 = ptrtoint ptr %3751 to i64
  %3753 = add i64 %3752, 63
  %3754 = and i64 %3753, -64
  %3755 = inttoptr i64 %3754 to ptr
  store double %3730, ptr %3755, align 64, !tbaa !7
  %3756 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 80)
  %3757 = ptrtoint ptr %3756 to i64
  %3758 = add i64 %3757, 63
  %3759 = and i64 %3758, -64
  %3760 = inttoptr i64 %3759 to ptr
  %3761 = load double, ptr %3750, align 64, !tbaa !7
  store double %3761, ptr %3760, align 64, !tbaa !7
  %3762 = load double, ptr %3755, align 64, !tbaa !7
  %3763 = getelementptr inbounds nuw i8, ptr %3760, i64 8
  store double %3762, ptr %3763, align 8, !tbaa !7
  tail call void @_mlir_memref_to_llvm_free(ptr %3751)
  tail call void @_mlir_memref_to_llvm_free(ptr %3746)
  %3764 = icmp eq ptr %3732, inttoptr (i64 3735928559 to ptr)
  br i1 %3764, label %3765, label %3767

3765:                                             ; preds = %3734
  %3766 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 384)
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(384) %3766, ptr noundef nonnull align 1 dereferenceable(384) inttoptr (i64 3735928559 to ptr), i64 384, i1 false)
  br label %3767

3767:                                             ; preds = %3765, %3734
  %.pn318 = phi ptr [ %3766, %3765 ], [ %3732, %3734 ]
  %3768 = icmp eq ptr %3735, inttoptr (i64 3735928559 to ptr)
  br i1 %3768, label %3769, label %3772

3769:                                             ; preds = %3767
  %3770 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 4)
  %3771 = load i32, ptr %3739, align 64
  store i32 %3771, ptr %3770, align 1
  br label %3772

3772:                                             ; preds = %3769, %3767
  %.pn249 = phi ptr [ %3770, %3769 ], [ %3735, %3767 ]
  %.pn247 = phi ptr [ %3770, %3769 ], [ %3739, %3767 ]
  %3773 = icmp eq ptr %3740, inttoptr (i64 3735928559 to ptr)
  br i1 %3773, label %3774, label %3777

3774:                                             ; preds = %3772
  %3775 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 4)
  %3776 = load i32, ptr %3744, align 64
  store i32 %3776, ptr %3775, align 1
  br label %3777

3777:                                             ; preds = %3774, %3772
  %.pn255 = phi ptr [ %3775, %3774 ], [ %3740, %3772 ]
  %.pn253 = phi ptr [ %3775, %3774 ], [ %3744, %3772 ]
  %3778 = icmp eq ptr %15, inttoptr (i64 3735928559 to ptr)
  br i1 %3778, label %3779, label %3783

3779:                                             ; preds = %3777
  %3780 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 4)
  %3781 = getelementptr inbounds i32, ptr %16, i64 %17
  %3782 = load i32, ptr %3781, align 1
  store i32 %3782, ptr %3780, align 1
  br label %3783

3783:                                             ; preds = %3779, %3777
  %.pn261 = phi ptr [ %3780, %3779 ], [ %15, %3777 ]
  %.pn259 = phi ptr [ %3780, %3779 ], [ %16, %3777 ]
  %.pn257 = phi i64 [ 0, %3779 ], [ %17, %3777 ]
  %3784 = icmp eq ptr %449, inttoptr (i64 3735928559 to ptr)
  br i1 %3784, label %3785, label %3788

3785:                                             ; preds = %3783
  %3786 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 8)
  %3787 = load i64, ptr %453, align 64
  store i64 %3787, ptr %3786, align 1
  br label %3788

3788:                                             ; preds = %3785, %3783
  %.pn271 = phi ptr [ %3786, %3785 ], [ %449, %3783 ]
  %.pn269 = phi ptr [ %3786, %3785 ], [ %453, %3783 ]
  %3789 = icmp eq ptr %3756, inttoptr (i64 3735928559 to ptr)
  br i1 %3789, label %3790, label %3792

3790:                                             ; preds = %3788
  %3791 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 16)
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(16) %3791, ptr noundef nonnull align 64 dereferenceable(16) %3760, i64 16, i1 false)
  br label %3792

3792:                                             ; preds = %3790, %3788
  %.pn281 = phi ptr [ %3791, %3790 ], [ %3756, %3788 ]
  %.pn279 = phi ptr [ %3791, %3790 ], [ %3760, %3788 ]
  %.pn317 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } poison, ptr %.pn318, 0
  %.pn315 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %.pn317, ptr %.pn318, 1
  %.pn313 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %.pn315, i64 0, 2
  %.pn311 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %.pn313, i64 4, 3, 0
  %.pn309 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %.pn311, i64 8, 3, 1
  %.pn307 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %.pn309, i64 3, 3, 2
  %.pn305 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %.pn307, i64 24, 4, 0
  %.pn303 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %.pn305, i64 3, 4, 1
  %3793 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %.pn303, i64 1, 4, 2
  %.pn278 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %.pn281, 0
  %.pn276 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %.pn278, ptr %.pn279, 1
  %.pn274 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %.pn276, i64 0, 2
  %.pn272 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %.pn274, i64 2, 3, 0
  %3794 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %.pn272, i64 1, 4, 0
  %.pn268 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %.pn271, 0
  %.pn266 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %.pn268, ptr %.pn269, 1
  %.pn264 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %.pn266, i64 0, 2
  %.pn262 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %.pn264, i64 2, 3, 0
  %3795 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %.pn262, i64 1, 4, 0
  %.pn258 = insertvalue { ptr, ptr, i64 } poison, ptr %.pn261, 0
  %.pn256 = insertvalue { ptr, ptr, i64 } %.pn258, ptr %.pn259, 1
  %3796 = insertvalue { ptr, ptr, i64 } %.pn256, i64 %.pn257, 2
  %.pn252 = insertvalue { ptr, ptr, i64 } poison, ptr %.pn255, 0
  %.pn250 = insertvalue { ptr, ptr, i64 } %.pn252, ptr %.pn253, 1
  %3797 = insertvalue { ptr, ptr, i64 } %.pn250, i64 0, 2
  %.pn246 = insertvalue { ptr, ptr, i64 } poison, ptr %.pn249, 0
  %.pn244 = insertvalue { ptr, ptr, i64 } %.pn246, ptr %.pn247, 1
  %3798 = insertvalue { ptr, ptr, i64 } %.pn244, i64 0, 2
  %3799 = insertvalue { { ptr, ptr, i64, [3 x i64], [3 x i64] }, { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64, [1 x i64], [1 x i64] }, { ptr, ptr, i64, [1 x i64], [1 x i64] } } poison, { ptr, ptr, i64, [3 x i64], [3 x i64] } %3793, 0
  %3800 = insertvalue { { ptr, ptr, i64, [3 x i64], [3 x i64] }, { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64, [1 x i64], [1 x i64] }, { ptr, ptr, i64, [1 x i64], [1 x i64] } } %3799, { ptr, ptr, i64 } %3798, 1
  %3801 = insertvalue { { ptr, ptr, i64, [3 x i64], [3 x i64] }, { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64, [1 x i64], [1 x i64] }, { ptr, ptr, i64, [1 x i64], [1 x i64] } } %3800, { ptr, ptr, i64 } %3797, 2
  %3802 = insertvalue { { ptr, ptr, i64, [3 x i64], [3 x i64] }, { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64, [1 x i64], [1 x i64] }, { ptr, ptr, i64, [1 x i64], [1 x i64] } } %3801, { ptr, ptr, i64 } %3796, 3
  %3803 = insertvalue { { ptr, ptr, i64, [3 x i64], [3 x i64] }, { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64, [1 x i64], [1 x i64] }, { ptr, ptr, i64, [1 x i64], [1 x i64] } } %3802, { ptr, ptr, i64, [1 x i64], [1 x i64] } %3795, 4
  %3804 = insertvalue { { ptr, ptr, i64, [3 x i64], [3 x i64] }, { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64, [1 x i64], [1 x i64] }, { ptr, ptr, i64, [1 x i64], [1 x i64] } } %3803, { ptr, ptr, i64, [1 x i64], [1 x i64] } %3794, 5
  ret { { ptr, ptr, i64, [3 x i64], [3 x i64] }, { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64, [1 x i64], [1 x i64] }, { ptr, ptr, i64, [1 x i64], [1 x i64] } } %3804
}

define void @_catalyst_pyface_jit_train_epoch_compiled(ptr writeonly captures(none) initializes((0, 224)) %0, ptr readonly captures(none) %1) local_unnamed_addr {
  %.unpack = load ptr, ptr %1, align 8
  %.elt1 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %.unpack2 = load ptr, ptr %.elt1, align 8
  %.elt3 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %.unpack4 = load ptr, ptr %.elt3, align 8
  %.elt5 = getelementptr inbounds nuw i8, ptr %1, i64 24
  %.unpack6 = load ptr, ptr %.elt5, align 8
  %.elt7 = getelementptr inbounds nuw i8, ptr %1, i64 32
  %.unpack8 = load ptr, ptr %.elt7, align 8
  %.elt9 = getelementptr inbounds nuw i8, ptr %1, i64 40
  %.unpack10 = load ptr, ptr %.elt9, align 8
  %.elt11 = getelementptr inbounds nuw i8, ptr %1, i64 48
  %.unpack12 = load ptr, ptr %.elt11, align 8
  %.elt13 = getelementptr inbounds nuw i8, ptr %1, i64 56
  %.unpack14 = load ptr, ptr %.elt13, align 8
  tail call void @_catalyst_ciface_jit_train_epoch_compiled(ptr %0, ptr %.unpack, ptr %.unpack2, ptr %.unpack4, ptr %.unpack6, ptr %.unpack8, ptr %.unpack10, ptr %.unpack12, ptr %.unpack14)
  ret void
}

define void @_catalyst_ciface_jit_train_epoch_compiled(ptr writeonly captures(none) initializes((0, 224)) %0, ptr readonly captures(none) %1, ptr readonly captures(none) %2, ptr readonly captures(none) %3, ptr readonly captures(none) %4, ptr readonly captures(none) %5, ptr readonly captures(none) %6, ptr readonly captures(none) %7, ptr readonly captures(none) %8) local_unnamed_addr {
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
  %.elt20 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %.unpack21 = load ptr, ptr %.elt20, align 8
  %.elt25 = getelementptr inbounds nuw i8, ptr %3, i64 8
  %.unpack26 = load ptr, ptr %.elt25, align 8
  %.unpack29 = load ptr, ptr %4, align 8
  %.elt30 = getelementptr inbounds nuw i8, ptr %4, i64 8
  %.unpack31 = load ptr, ptr %.elt30, align 8
  %.elt32 = getelementptr inbounds nuw i8, ptr %4, i64 16
  %.unpack33 = load i64, ptr %.elt32, align 8
  %.elt35 = getelementptr inbounds nuw i8, ptr %5, i64 8
  %.unpack36 = load ptr, ptr %.elt35, align 8
  %.unpack45 = load ptr, ptr %6, align 8
  %.elt46 = getelementptr inbounds nuw i8, ptr %6, i64 8
  %.unpack47 = load ptr, ptr %.elt46, align 8
  %.elt61 = getelementptr inbounds nuw i8, ptr %7, i64 8
  %.unpack62 = load ptr, ptr %.elt61, align 8
  %.elt72 = getelementptr inbounds nuw i8, ptr %8, i64 8
  %.unpack73 = load ptr, ptr %.elt72, align 8
  %10 = tail call { { ptr, ptr, i64, [3 x i64], [3 x i64] }, { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64, [1 x i64], [1 x i64] }, { ptr, ptr, i64, [1 x i64], [1 x i64] } } @jit_train_epoch_compiled(ptr poison, ptr %.unpack2, i64 %.unpack4, i64 %.unpack6.unpack, i64 %.unpack6.unpack10, i64 %.unpack6.unpack12, i64 poison, i64 poison, i64 poison, ptr poison, ptr %.unpack21, i64 poison, ptr poison, ptr %.unpack26, i64 poison, ptr %.unpack29, ptr %.unpack31, i64 %.unpack33, ptr poison, ptr %.unpack36, i64 poison, i64 poison, i64 poison, ptr %.unpack45, ptr %.unpack47, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, ptr poison, ptr %.unpack62, i64 poison, i64 poison, i64 poison, ptr poison, ptr %.unpack73, i64 poison, i64 poison, i64 poison)
  %.elt = extractvalue { { ptr, ptr, i64, [3 x i64], [3 x i64] }, { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64, [1 x i64], [1 x i64] }, { ptr, ptr, i64, [1 x i64], [1 x i64] } } %10, 0
  %.elt.elt = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %.elt, 0
  store ptr %.elt.elt, ptr %0, align 8
  %.repack92 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %.elt.elt93 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %.elt, 1
  store ptr %.elt.elt93, ptr %.repack92, align 8
  %.repack94 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %.elt.elt95 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %.elt, 2
  store i64 %.elt.elt95, ptr %.repack94, align 8
  %.repack96 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %.elt.elt97 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %.elt, 3
  %.elt.elt97.elt = extractvalue [3 x i64] %.elt.elt97, 0
  store i64 %.elt.elt97.elt, ptr %.repack96, align 8
  %.repack96.repack100 = getelementptr inbounds nuw i8, ptr %0, i64 32
  %.elt.elt97.elt101 = extractvalue [3 x i64] %.elt.elt97, 1
  store i64 %.elt.elt97.elt101, ptr %.repack96.repack100, align 8
  %.repack96.repack102 = getelementptr inbounds nuw i8, ptr %0, i64 40
  %.elt.elt97.elt103 = extractvalue [3 x i64] %.elt.elt97, 2
  store i64 %.elt.elt97.elt103, ptr %.repack96.repack102, align 8
  %.repack98 = getelementptr inbounds nuw i8, ptr %0, i64 48
  %.elt.elt99 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %.elt, 4
  %.elt.elt99.elt = extractvalue [3 x i64] %.elt.elt99, 0
  store i64 %.elt.elt99.elt, ptr %.repack98, align 8
  %.repack98.repack104 = getelementptr inbounds nuw i8, ptr %0, i64 56
  %.elt.elt99.elt105 = extractvalue [3 x i64] %.elt.elt99, 1
  store i64 %.elt.elt99.elt105, ptr %.repack98.repack104, align 8
  %.repack98.repack106 = getelementptr inbounds nuw i8, ptr %0, i64 64
  %.elt.elt99.elt107 = extractvalue [3 x i64] %.elt.elt99, 2
  store i64 %.elt.elt99.elt107, ptr %.repack98.repack106, align 8
  %.repack82 = getelementptr inbounds nuw i8, ptr %0, i64 72
  %.elt83 = extractvalue { { ptr, ptr, i64, [3 x i64], [3 x i64] }, { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64, [1 x i64], [1 x i64] }, { ptr, ptr, i64, [1 x i64], [1 x i64] } } %10, 1
  %.elt83.elt = extractvalue { ptr, ptr, i64 } %.elt83, 0
  store ptr %.elt83.elt, ptr %.repack82, align 8
  %.repack82.repack108 = getelementptr inbounds nuw i8, ptr %0, i64 80
  %.elt83.elt109 = extractvalue { ptr, ptr, i64 } %.elt83, 1
  store ptr %.elt83.elt109, ptr %.repack82.repack108, align 8
  %.repack82.repack110 = getelementptr inbounds nuw i8, ptr %0, i64 88
  %.elt83.elt111 = extractvalue { ptr, ptr, i64 } %.elt83, 2
  store i64 %.elt83.elt111, ptr %.repack82.repack110, align 8
  %.repack84 = getelementptr inbounds nuw i8, ptr %0, i64 96
  %.elt85 = extractvalue { { ptr, ptr, i64, [3 x i64], [3 x i64] }, { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64, [1 x i64], [1 x i64] }, { ptr, ptr, i64, [1 x i64], [1 x i64] } } %10, 2
  %.elt85.elt = extractvalue { ptr, ptr, i64 } %.elt85, 0
  store ptr %.elt85.elt, ptr %.repack84, align 8
  %.repack84.repack112 = getelementptr inbounds nuw i8, ptr %0, i64 104
  %.elt85.elt113 = extractvalue { ptr, ptr, i64 } %.elt85, 1
  store ptr %.elt85.elt113, ptr %.repack84.repack112, align 8
  %.repack84.repack114 = getelementptr inbounds nuw i8, ptr %0, i64 112
  %.elt85.elt115 = extractvalue { ptr, ptr, i64 } %.elt85, 2
  store i64 %.elt85.elt115, ptr %.repack84.repack114, align 8
  %.repack86 = getelementptr inbounds nuw i8, ptr %0, i64 120
  %.elt87 = extractvalue { { ptr, ptr, i64, [3 x i64], [3 x i64] }, { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64, [1 x i64], [1 x i64] }, { ptr, ptr, i64, [1 x i64], [1 x i64] } } %10, 3
  %.elt87.elt = extractvalue { ptr, ptr, i64 } %.elt87, 0
  store ptr %.elt87.elt, ptr %.repack86, align 8
  %.repack86.repack116 = getelementptr inbounds nuw i8, ptr %0, i64 128
  %.elt87.elt117 = extractvalue { ptr, ptr, i64 } %.elt87, 1
  store ptr %.elt87.elt117, ptr %.repack86.repack116, align 8
  %.repack86.repack118 = getelementptr inbounds nuw i8, ptr %0, i64 136
  %.elt87.elt119 = extractvalue { ptr, ptr, i64 } %.elt87, 2
  store i64 %.elt87.elt119, ptr %.repack86.repack118, align 8
  %.repack88 = getelementptr inbounds nuw i8, ptr %0, i64 144
  %.elt89 = extractvalue { { ptr, ptr, i64, [3 x i64], [3 x i64] }, { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64, [1 x i64], [1 x i64] }, { ptr, ptr, i64, [1 x i64], [1 x i64] } } %10, 4
  %.elt89.elt = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %.elt89, 0
  store ptr %.elt89.elt, ptr %.repack88, align 8
  %.repack88.repack120 = getelementptr inbounds nuw i8, ptr %0, i64 152
  %.elt89.elt121 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %.elt89, 1
  store ptr %.elt89.elt121, ptr %.repack88.repack120, align 8
  %.repack88.repack122 = getelementptr inbounds nuw i8, ptr %0, i64 160
  %.elt89.elt123 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %.elt89, 2
  store i64 %.elt89.elt123, ptr %.repack88.repack122, align 8
  %.repack88.repack124 = getelementptr inbounds nuw i8, ptr %0, i64 168
  %.elt89.elt125 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %.elt89, 3
  %11 = extractvalue [1 x i64] %.elt89.elt125, 0
  store i64 %11, ptr %.repack88.repack124, align 8
  %.repack88.repack126 = getelementptr inbounds nuw i8, ptr %0, i64 176
  %.elt89.elt127 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %.elt89, 4
  %12 = extractvalue [1 x i64] %.elt89.elt127, 0
  store i64 %12, ptr %.repack88.repack126, align 8
  %.repack90 = getelementptr inbounds nuw i8, ptr %0, i64 184
  %.elt91 = extractvalue { { ptr, ptr, i64, [3 x i64], [3 x i64] }, { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64, [1 x i64], [1 x i64] }, { ptr, ptr, i64, [1 x i64], [1 x i64] } } %10, 5
  %.elt91.elt = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %.elt91, 0
  store ptr %.elt91.elt, ptr %.repack90, align 8
  %.repack90.repack128 = getelementptr inbounds nuw i8, ptr %0, i64 192
  %.elt91.elt129 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %.elt91, 1
  store ptr %.elt91.elt129, ptr %.repack90.repack128, align 8
  %.repack90.repack130 = getelementptr inbounds nuw i8, ptr %0, i64 200
  %.elt91.elt131 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %.elt91, 2
  store i64 %.elt91.elt131, ptr %.repack90.repack130, align 8
  %.repack90.repack132 = getelementptr inbounds nuw i8, ptr %0, i64 208
  %.elt91.elt133 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %.elt91, 3
  %13 = extractvalue [1 x i64] %.elt91.elt133, 0
  store i64 %13, ptr %.repack90.repack132, align 8
  %.repack90.repack134 = getelementptr inbounds nuw i8, ptr %0, i64 216
  %.elt91.elt135 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %.elt91, 4
  %14 = extractvalue [1 x i64] %.elt91.elt135, 0
  store i64 %14, ptr %.repack90.repack134, align 8
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
  tail call void @__catalyst__rt__device_init(ptr nonnull @"/home/pichau/QAgents/.venv/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.so", ptr nonnull @LightningSimulator, ptr nonnull @"{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}", i64 0, i1 false)
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
  %16 = load double, ptr %.unpack45, align 8, !tbaa !7
  %17 = getelementptr inbounds nuw double, ptr %11, i64 %15
  %18 = load double, ptr %17, align 8, !tbaa !7
  %19 = getelementptr inbounds nuw double, ptr %.unpack34, i64 %15
  %20 = load double, ptr %19, align 8, !tbaa !7
  %21 = fmul double %16, %18
  %22 = fadd double %20, %21
  store double %22, ptr %19, align 8, !tbaa !7
  %23 = add nuw nsw i64 %15, 1
  %exitcond.not = icmp eq i64 %23, %.unpack38.unpack
  br i1 %exitcond.not, label %._crit_edge, label %.lr.ph

._crit_edge:                                      ; preds = %.lr.ph, %9
  ret void
}

; Function Attrs: noinline
define void @qnode_forward_0.quantum(ptr %0, ptr %1, ptr %2, ptr %3) local_unnamed_addr #4 !enzyme_augment !85 !enzyme_gradient !86 {
  %5 = load volatile { ptr, ptr, i64, [3 x i64], [3 x i64] }, ptr %0, align 8
  %6 = load volatile { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %1, align 8
  %7 = load volatile { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %2, align 8
  %8 = load volatile { ptr, ptr, i64 }, ptr %3, align 8
  tail call void @__catalyst__rt__device_init(ptr nonnull @"/home/pichau/QAgents/.venv/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.so", ptr nonnull @LightningSimulator, ptr nonnull @"{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}", i64 0, i1 false)
  %9 = tail call ptr @__catalyst__rt__qubit_allocate_array(i64 8)
  %10 = tail call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %9, i64 7)
  %11 = load ptr, ptr %10, align 8
  %12 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %13 = load double, ptr %12, align 8, !tbaa !7
  tail call void @__catalyst__qis__RY(double %13, ptr %11, ptr null)
  %14 = getelementptr inbounds nuw i8, ptr %12, i64 8
  %15 = load double, ptr %14, align 8, !tbaa !7
  tail call void @__catalyst__qis__RZ(double %15, ptr %11, ptr null)
  %16 = getelementptr inbounds nuw i8, ptr %12, i64 16
  %17 = load double, ptr %16, align 8, !tbaa !7
  tail call void @__catalyst__qis__RY(double %17, ptr %11, ptr null)
  %18 = getelementptr inbounds nuw i8, ptr %12, i64 24
  %19 = load double, ptr %18, align 8, !tbaa !7
  tail call void @__catalyst__qis__RZ(double %19, ptr %11, ptr null)
  %20 = tail call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %9, i64 6)
  %21 = load ptr, ptr %20, align 8
  %22 = getelementptr inbounds nuw i8, ptr %12, i64 32
  %23 = load double, ptr %22, align 8, !tbaa !7
  tail call void @__catalyst__qis__RY(double %23, ptr %21, ptr null)
  %24 = getelementptr inbounds nuw i8, ptr %12, i64 40
  %25 = load double, ptr %24, align 8, !tbaa !7
  tail call void @__catalyst__qis__RZ(double %25, ptr %21, ptr null)
  %26 = getelementptr inbounds nuw i8, ptr %12, i64 48
  %27 = load double, ptr %26, align 8, !tbaa !7
  tail call void @__catalyst__qis__RY(double %27, ptr %21, ptr null)
  %28 = getelementptr inbounds nuw i8, ptr %12, i64 56
  %29 = load double, ptr %28, align 8, !tbaa !7
  tail call void @__catalyst__qis__RZ(double %29, ptr %21, ptr null)
  %30 = tail call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %9, i64 5)
  %31 = load ptr, ptr %30, align 8
  %32 = getelementptr inbounds nuw i8, ptr %12, i64 64
  %33 = load double, ptr %32, align 8, !tbaa !7
  tail call void @__catalyst__qis__RY(double %33, ptr %31, ptr null)
  %34 = getelementptr inbounds nuw i8, ptr %12, i64 72
  %35 = load double, ptr %34, align 8, !tbaa !7
  tail call void @__catalyst__qis__RZ(double %35, ptr %31, ptr null)
  %36 = getelementptr inbounds nuw i8, ptr %12, i64 80
  %37 = load double, ptr %36, align 8, !tbaa !7
  tail call void @__catalyst__qis__RY(double %37, ptr %31, ptr null)
  %38 = getelementptr inbounds nuw i8, ptr %12, i64 88
  %39 = load double, ptr %38, align 8, !tbaa !7
  tail call void @__catalyst__qis__RZ(double %39, ptr %31, ptr null)
  %40 = tail call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %9, i64 4)
  %41 = load ptr, ptr %40, align 8
  %42 = getelementptr inbounds nuw i8, ptr %12, i64 96
  %43 = load double, ptr %42, align 8, !tbaa !7
  tail call void @__catalyst__qis__RY(double %43, ptr %41, ptr null)
  %44 = getelementptr inbounds nuw i8, ptr %12, i64 104
  %45 = load double, ptr %44, align 8, !tbaa !7
  tail call void @__catalyst__qis__RZ(double %45, ptr %41, ptr null)
  %46 = getelementptr inbounds nuw i8, ptr %12, i64 112
  %47 = load double, ptr %46, align 8, !tbaa !7
  tail call void @__catalyst__qis__RY(double %47, ptr %41, ptr null)
  %48 = getelementptr inbounds nuw i8, ptr %12, i64 120
  %49 = load double, ptr %48, align 8, !tbaa !7
  tail call void @__catalyst__qis__RZ(double %49, ptr %41, ptr null)
  %50 = tail call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %9, i64 3)
  %51 = load ptr, ptr %50, align 8
  %52 = getelementptr inbounds nuw i8, ptr %12, i64 128
  %53 = load double, ptr %52, align 8, !tbaa !7
  tail call void @__catalyst__qis__RY(double %53, ptr %51, ptr null)
  %54 = getelementptr inbounds nuw i8, ptr %12, i64 136
  %55 = load double, ptr %54, align 8, !tbaa !7
  tail call void @__catalyst__qis__RZ(double %55, ptr %51, ptr null)
  %56 = getelementptr inbounds nuw i8, ptr %12, i64 144
  %57 = load double, ptr %56, align 8, !tbaa !7
  tail call void @__catalyst__qis__RY(double %57, ptr %51, ptr null)
  %58 = getelementptr inbounds nuw i8, ptr %12, i64 152
  %59 = load double, ptr %58, align 8, !tbaa !7
  tail call void @__catalyst__qis__RZ(double %59, ptr %51, ptr null)
  %60 = tail call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %9, i64 2)
  %61 = load ptr, ptr %60, align 8
  %62 = getelementptr inbounds nuw i8, ptr %12, i64 160
  %63 = load double, ptr %62, align 8, !tbaa !7
  tail call void @__catalyst__qis__RY(double %63, ptr %61, ptr null)
  %64 = getelementptr inbounds nuw i8, ptr %12, i64 168
  %65 = load double, ptr %64, align 8, !tbaa !7
  tail call void @__catalyst__qis__RZ(double %65, ptr %61, ptr null)
  %66 = getelementptr inbounds nuw i8, ptr %12, i64 176
  %67 = load double, ptr %66, align 8, !tbaa !7
  tail call void @__catalyst__qis__RY(double %67, ptr %61, ptr null)
  %68 = getelementptr inbounds nuw i8, ptr %12, i64 184
  %69 = load double, ptr %68, align 8, !tbaa !7
  tail call void @__catalyst__qis__RZ(double %69, ptr %61, ptr null)
  %70 = tail call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %9, i64 0)
  %71 = load ptr, ptr %70, align 8
  %72 = getelementptr inbounds nuw i8, ptr %12, i64 192
  %73 = load double, ptr %72, align 8, !tbaa !7
  tail call void @__catalyst__qis__RY(double %73, ptr %71, ptr null)
  %74 = getelementptr inbounds nuw i8, ptr %12, i64 200
  %75 = load double, ptr %74, align 8, !tbaa !7
  tail call void @__catalyst__qis__RZ(double %75, ptr %71, ptr null)
  %76 = getelementptr inbounds nuw i8, ptr %12, i64 208
  %77 = load double, ptr %76, align 8, !tbaa !7
  tail call void @__catalyst__qis__RY(double %77, ptr %71, ptr null)
  %78 = getelementptr inbounds nuw i8, ptr %12, i64 216
  %79 = load double, ptr %78, align 8, !tbaa !7
  tail call void @__catalyst__qis__RZ(double %79, ptr %71, ptr null)
  %80 = tail call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %9, i64 1)
  %81 = load ptr, ptr %80, align 8
  %82 = getelementptr inbounds nuw i8, ptr %12, i64 224
  %83 = load double, ptr %82, align 8, !tbaa !7
  tail call void @__catalyst__qis__RY(double %83, ptr %81, ptr null)
  %84 = getelementptr inbounds nuw i8, ptr %12, i64 232
  %85 = load double, ptr %84, align 8, !tbaa !7
  tail call void @__catalyst__qis__RZ(double %85, ptr %81, ptr null)
  %86 = getelementptr inbounds nuw i8, ptr %12, i64 240
  %87 = load double, ptr %86, align 8, !tbaa !7
  tail call void @__catalyst__qis__RY(double %87, ptr %81, ptr null)
  %88 = getelementptr inbounds nuw i8, ptr %12, i64 248
  %89 = load double, ptr %88, align 8, !tbaa !7
  tail call void @__catalyst__qis__RZ(double %89, ptr %81, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %71, ptr %81, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %81, ptr %61, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %61, ptr %51, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %51, ptr %41, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %41, ptr %31, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %31, ptr %21, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %21, ptr %11, ptr null)
  %90 = getelementptr inbounds nuw i8, ptr %12, i64 256
  %91 = load double, ptr %90, align 8, !tbaa !7
  tail call void @__catalyst__qis__RZ(double %91, ptr %21, ptr null)
  %92 = getelementptr inbounds nuw i8, ptr %12, i64 264
  %93 = load double, ptr %92, align 8, !tbaa !7
  tail call void @__catalyst__qis__RY(double %93, ptr %21, ptr null)
  %94 = getelementptr inbounds nuw i8, ptr %12, i64 272
  %95 = load double, ptr %94, align 8, !tbaa !7
  tail call void @__catalyst__qis__RZ(double %95, ptr %21, ptr null)
  %96 = getelementptr inbounds nuw i8, ptr %12, i64 280
  %97 = load double, ptr %96, align 8, !tbaa !7
  tail call void @__catalyst__qis__RZ(double %97, ptr %41, ptr null)
  %98 = getelementptr inbounds nuw i8, ptr %12, i64 288
  %99 = load double, ptr %98, align 8, !tbaa !7
  tail call void @__catalyst__qis__RY(double %99, ptr %41, ptr null)
  %100 = getelementptr inbounds nuw i8, ptr %12, i64 296
  %101 = load double, ptr %100, align 8, !tbaa !7
  tail call void @__catalyst__qis__RZ(double %101, ptr %41, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %11, ptr %71, ptr null)
  %102 = getelementptr inbounds nuw i8, ptr %12, i64 304
  %103 = load double, ptr %102, align 8, !tbaa !7
  tail call void @__catalyst__qis__RZ(double %103, ptr %71, ptr null)
  %104 = getelementptr inbounds nuw i8, ptr %12, i64 312
  %105 = load double, ptr %104, align 8, !tbaa !7
  tail call void @__catalyst__qis__RY(double %105, ptr %71, ptr null)
  %106 = getelementptr inbounds nuw i8, ptr %12, i64 320
  %107 = load double, ptr %106, align 8, !tbaa !7
  tail call void @__catalyst__qis__RZ(double %107, ptr %71, ptr null)
  %108 = getelementptr inbounds nuw i8, ptr %12, i64 328
  %109 = load double, ptr %108, align 8, !tbaa !7
  tail call void @__catalyst__qis__RZ(double %109, ptr %61, ptr null)
  %110 = getelementptr inbounds nuw i8, ptr %12, i64 336
  %111 = load double, ptr %110, align 8, !tbaa !7
  tail call void @__catalyst__qis__RY(double %111, ptr %61, ptr null)
  %112 = getelementptr inbounds nuw i8, ptr %12, i64 344
  %113 = load double, ptr %112, align 8, !tbaa !7
  tail call void @__catalyst__qis__RZ(double %113, ptr %61, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %71, ptr %61, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %61, ptr %41, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %41, ptr %21, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %21, ptr %71, ptr null)
  %114 = getelementptr inbounds nuw i8, ptr %12, i64 352
  %115 = load double, ptr %114, align 8, !tbaa !7
  tail call void @__catalyst__qis__RZ(double %115, ptr %71, ptr null)
  %116 = getelementptr inbounds nuw i8, ptr %12, i64 360
  %117 = load double, ptr %116, align 8, !tbaa !7
  tail call void @__catalyst__qis__RY(double %117, ptr %71, ptr null)
  %118 = getelementptr inbounds nuw i8, ptr %12, i64 368
  %119 = load double, ptr %118, align 8, !tbaa !7
  tail call void @__catalyst__qis__RZ(double %119, ptr %71, ptr null)
  %120 = getelementptr inbounds nuw i8, ptr %12, i64 376
  %121 = load double, ptr %120, align 8, !tbaa !7
  tail call void @__catalyst__qis__RZ(double %121, ptr %31, ptr null)
  %122 = getelementptr inbounds nuw i8, ptr %12, i64 384
  %123 = load double, ptr %122, align 8, !tbaa !7
  tail call void @__catalyst__qis__RY(double %123, ptr %31, ptr null)
  %124 = getelementptr inbounds nuw i8, ptr %12, i64 392
  %125 = load double, ptr %124, align 8, !tbaa !7
  tail call void @__catalyst__qis__RZ(double %125, ptr %31, ptr null)
  %126 = getelementptr inbounds nuw i8, ptr %12, i64 400
  %127 = load double, ptr %126, align 8, !tbaa !7
  tail call void @__catalyst__qis__RZ(double %127, ptr %81, ptr null)
  %128 = getelementptr inbounds nuw i8, ptr %12, i64 408
  %129 = load double, ptr %128, align 8, !tbaa !7
  tail call void @__catalyst__qis__RY(double %129, ptr %81, ptr null)
  %130 = getelementptr inbounds nuw i8, ptr %12, i64 416
  %131 = load double, ptr %130, align 8, !tbaa !7
  tail call void @__catalyst__qis__RZ(double %131, ptr %81, ptr null)
  %132 = getelementptr inbounds nuw i8, ptr %12, i64 424
  %133 = load double, ptr %132, align 8, !tbaa !7
  tail call void @__catalyst__qis__RZ(double %133, ptr %51, ptr null)
  %134 = getelementptr inbounds nuw i8, ptr %12, i64 432
  %135 = load double, ptr %134, align 8, !tbaa !7
  tail call void @__catalyst__qis__RY(double %135, ptr %51, ptr null)
  %136 = getelementptr inbounds nuw i8, ptr %12, i64 440
  %137 = load double, ptr %136, align 8, !tbaa !7
  tail call void @__catalyst__qis__RZ(double %137, ptr %51, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %81, ptr %51, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %51, ptr %31, ptr null)
  %138 = getelementptr inbounds nuw i8, ptr %12, i64 448
  %139 = load double, ptr %138, align 8, !tbaa !7
  tail call void @__catalyst__qis__RZ(double %139, ptr %51, ptr null)
  %140 = getelementptr inbounds nuw i8, ptr %12, i64 456
  %141 = load double, ptr %140, align 8, !tbaa !7
  tail call void @__catalyst__qis__RY(double %141, ptr %51, ptr null)
  %142 = getelementptr inbounds nuw i8, ptr %12, i64 464
  %143 = load double, ptr %142, align 8, !tbaa !7
  tail call void @__catalyst__qis__RZ(double %143, ptr %51, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %71, ptr %51, ptr null)
  %144 = getelementptr inbounds nuw i8, ptr %12, i64 472
  %145 = load double, ptr %144, align 8, !tbaa !7
  tail call void @__catalyst__qis__RZ(double %145, ptr %61, ptr null)
  %146 = getelementptr inbounds nuw i8, ptr %12, i64 480
  %147 = load double, ptr %146, align 8, !tbaa !7
  tail call void @__catalyst__qis__RY(double %147, ptr %61, ptr null)
  %148 = getelementptr inbounds nuw i8, ptr %12, i64 488
  %149 = load double, ptr %148, align 8, !tbaa !7
  tail call void @__catalyst__qis__RZ(double %149, ptr %61, ptr null)
  %150 = getelementptr inbounds nuw i8, ptr %12, i64 496
  %151 = load double, ptr %150, align 8, !tbaa !7
  tail call void @__catalyst__qis__RZ(double %151, ptr %11, ptr null)
  %152 = getelementptr inbounds nuw i8, ptr %12, i64 504
  %153 = load double, ptr %152, align 8, !tbaa !7
  tail call void @__catalyst__qis__RY(double %153, ptr %11, ptr null)
  %154 = getelementptr inbounds nuw i8, ptr %12, i64 512
  %155 = load double, ptr %154, align 8, !tbaa !7
  tail call void @__catalyst__qis__RZ(double %155, ptr %11, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %31, ptr %11, ptr null)
  %156 = getelementptr inbounds nuw i8, ptr %12, i64 520
  %157 = load double, ptr %156, align 8, !tbaa !7
  tail call void @__catalyst__qis__RZ(double %157, ptr %31, ptr null)
  %158 = getelementptr inbounds nuw i8, ptr %12, i64 528
  %159 = load double, ptr %158, align 8, !tbaa !7
  tail call void @__catalyst__qis__RY(double %159, ptr %31, ptr null)
  %160 = getelementptr inbounds nuw i8, ptr %12, i64 536
  %161 = load double, ptr %160, align 8, !tbaa !7
  tail call void @__catalyst__qis__RZ(double %161, ptr %31, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %61, ptr %31, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %31, ptr %71, ptr null)
  %162 = getelementptr inbounds nuw i8, ptr %12, i64 544
  %163 = load double, ptr %162, align 8, !tbaa !7
  tail call void @__catalyst__qis__RZ(double %163, ptr %71, ptr null)
  %164 = getelementptr inbounds nuw i8, ptr %12, i64 552
  %165 = load double, ptr %164, align 8, !tbaa !7
  tail call void @__catalyst__qis__RY(double %165, ptr %71, ptr null)
  %166 = getelementptr inbounds nuw i8, ptr %12, i64 560
  %167 = load double, ptr %166, align 8, !tbaa !7
  tail call void @__catalyst__qis__RZ(double %167, ptr %71, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %11, ptr %81, ptr null)
  %168 = getelementptr inbounds nuw i8, ptr %12, i64 568
  %169 = load double, ptr %168, align 8, !tbaa !7
  tail call void @__catalyst__qis__RZ(double %169, ptr %11, ptr null)
  %170 = getelementptr inbounds nuw i8, ptr %12, i64 576
  %171 = load double, ptr %170, align 8, !tbaa !7
  tail call void @__catalyst__qis__RY(double %171, ptr %11, ptr null)
  %172 = getelementptr inbounds nuw i8, ptr %12, i64 584
  %173 = load double, ptr %172, align 8, !tbaa !7
  tail call void @__catalyst__qis__RZ(double %173, ptr %11, ptr null)
  %174 = getelementptr inbounds nuw i8, ptr %12, i64 592
  %175 = load double, ptr %174, align 8, !tbaa !7
  tail call void @__catalyst__qis__RZ(double %175, ptr %81, ptr null)
  %176 = getelementptr inbounds nuw i8, ptr %12, i64 600
  %177 = load double, ptr %176, align 8, !tbaa !7
  tail call void @__catalyst__qis__RY(double %177, ptr %81, ptr null)
  %178 = getelementptr inbounds nuw i8, ptr %12, i64 608
  %179 = load double, ptr %178, align 8, !tbaa !7
  tail call void @__catalyst__qis__RZ(double %179, ptr %81, ptr null)
  %180 = getelementptr inbounds nuw i8, ptr %12, i64 616
  %181 = load double, ptr %180, align 8, !tbaa !7
  tail call void @__catalyst__qis__RZ(double %181, ptr %41, ptr null)
  %182 = getelementptr inbounds nuw i8, ptr %12, i64 624
  %183 = load double, ptr %182, align 8, !tbaa !7
  tail call void @__catalyst__qis__RY(double %183, ptr %41, ptr null)
  %184 = getelementptr inbounds nuw i8, ptr %12, i64 632
  %185 = load double, ptr %184, align 8, !tbaa !7
  tail call void @__catalyst__qis__RZ(double %185, ptr %41, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %81, ptr %41, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %41, ptr %11, ptr null)
  %186 = getelementptr inbounds nuw i8, ptr %12, i64 640
  %187 = load double, ptr %186, align 8, !tbaa !7
  tail call void @__catalyst__qis__RZ(double %187, ptr %41, ptr null)
  %188 = getelementptr inbounds nuw i8, ptr %12, i64 648
  %189 = load double, ptr %188, align 8, !tbaa !7
  tail call void @__catalyst__qis__RY(double %189, ptr %41, ptr null)
  %190 = getelementptr inbounds nuw i8, ptr %12, i64 656
  %191 = load double, ptr %190, align 8, !tbaa !7
  tail call void @__catalyst__qis__RZ(double %191, ptr %41, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %71, ptr %41, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %41, ptr %71, ptr null)
  %192 = getelementptr inbounds nuw i8, ptr %12, i64 664
  %193 = load double, ptr %192, align 8, !tbaa !7
  tail call void @__catalyst__qis__RZ(double %193, ptr %21, ptr null)
  %194 = getelementptr inbounds nuw i8, ptr %12, i64 672
  %195 = load double, ptr %194, align 8, !tbaa !7
  tail call void @__catalyst__qis__RY(double %195, ptr %21, ptr null)
  %196 = getelementptr inbounds nuw i8, ptr %12, i64 680
  %197 = load double, ptr %196, align 8, !tbaa !7
  tail call void @__catalyst__qis__RZ(double %197, ptr %21, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %51, ptr %21, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %21, ptr %81, ptr null)
  %198 = getelementptr inbounds nuw i8, ptr %12, i64 688
  %199 = load double, ptr %198, align 8, !tbaa !7
  tail call void @__catalyst__qis__RZ(double %199, ptr %81, ptr null)
  %200 = getelementptr inbounds nuw i8, ptr %12, i64 696
  %201 = load double, ptr %200, align 8, !tbaa !7
  tail call void @__catalyst__qis__RY(double %201, ptr %81, ptr null)
  %202 = getelementptr inbounds nuw i8, ptr %12, i64 704
  %203 = load double, ptr %202, align 8, !tbaa !7
  tail call void @__catalyst__qis__RZ(double %203, ptr %81, ptr null)
  %204 = getelementptr inbounds nuw i8, ptr %12, i64 712
  %205 = load double, ptr %204, align 8, !tbaa !7
  tail call void @__catalyst__qis__RZ(double %205, ptr %31, ptr null)
  %206 = getelementptr inbounds nuw i8, ptr %12, i64 720
  %207 = load double, ptr %206, align 8, !tbaa !7
  tail call void @__catalyst__qis__RY(double %207, ptr %31, ptr null)
  %208 = getelementptr inbounds nuw i8, ptr %12, i64 728
  %209 = load double, ptr %208, align 8, !tbaa !7
  tail call void @__catalyst__qis__RZ(double %209, ptr %31, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %81, ptr %31, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %31, ptr %81, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %11, ptr %61, ptr null)
  %210 = getelementptr inbounds nuw i8, ptr %12, i64 736
  %211 = load double, ptr %210, align 8, !tbaa !7
  tail call void @__catalyst__qis__RZ(double %211, ptr %61, ptr null)
  %212 = getelementptr inbounds nuw i8, ptr %12, i64 744
  %213 = load double, ptr %212, align 8, !tbaa !7
  tail call void @__catalyst__qis__RY(double %213, ptr %61, ptr null)
  %214 = getelementptr inbounds nuw i8, ptr %12, i64 752
  %215 = load double, ptr %214, align 8, !tbaa !7
  tail call void @__catalyst__qis__RZ(double %215, ptr %61, ptr null)
  %216 = getelementptr inbounds nuw i8, ptr %12, i64 760
  %217 = load double, ptr %216, align 8, !tbaa !7
  tail call void @__catalyst__qis__RZ(double %217, ptr %21, ptr null)
  %218 = getelementptr inbounds nuw i8, ptr %12, i64 768
  %219 = load double, ptr %218, align 8, !tbaa !7
  tail call void @__catalyst__qis__RY(double %219, ptr %21, ptr null)
  %220 = getelementptr inbounds nuw i8, ptr %12, i64 776
  %221 = load double, ptr %220, align 8, !tbaa !7
  tail call void @__catalyst__qis__RZ(double %221, ptr %21, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %61, ptr %21, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %21, ptr %61, ptr null)
  %222 = getelementptr inbounds nuw i8, ptr %12, i64 784
  %223 = load double, ptr %222, align 8, !tbaa !7
  tail call void @__catalyst__qis__RZ(double %223, ptr %51, ptr null)
  %224 = getelementptr inbounds nuw i8, ptr %12, i64 792
  %225 = load double, ptr %224, align 8, !tbaa !7
  tail call void @__catalyst__qis__RY(double %225, ptr %51, ptr null)
  %226 = getelementptr inbounds nuw i8, ptr %12, i64 800
  %227 = load double, ptr %226, align 8, !tbaa !7
  tail call void @__catalyst__qis__RZ(double %227, ptr %51, ptr null)
  %228 = getelementptr inbounds nuw i8, ptr %12, i64 808
  %229 = load double, ptr %228, align 8, !tbaa !7
  tail call void @__catalyst__qis__RZ(double %229, ptr %11, ptr null)
  %230 = getelementptr inbounds nuw i8, ptr %12, i64 816
  %231 = load double, ptr %230, align 8, !tbaa !7
  tail call void @__catalyst__qis__RY(double %231, ptr %11, ptr null)
  %232 = getelementptr inbounds nuw i8, ptr %12, i64 824
  %233 = load double, ptr %232, align 8, !tbaa !7
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
  store double %235, ptr %240, align 64, !tbaa !7
  tail call void @__catalyst__rt__qubit_release_array(ptr %9)
  tail call void @__catalyst__rt__device_release()
  %241 = load double, ptr %240, align 64, !tbaa !7
  %242 = extractvalue { ptr, ptr, i64 } %8, 1
  store double %241, ptr %242, align 8, !tbaa !7
  ret void
}

define noalias noundef ptr @qnode_forward_0.quantum.augfwd(ptr %0, ptr readnone captures(none) %1, ptr %2, ptr readnone captures(none) %3, ptr %4, ptr readnone captures(none) %5, ptr %6, ptr readnone captures(none) %7) local_unnamed_addr #3 {
  tail call void @qnode_forward_0.quantum(ptr %0, ptr %2, ptr %4, ptr %6)
  ret ptr null
}

define { ptr, ptr, i64 } @qnode_forward_0.preprocess(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, ptr %9, ptr %10, i64 %11, i64 %12, i64 %13, i64 %14) local_unnamed_addr {
.preheader.preheader:
  %15 = alloca { ptr, ptr, i64 }, align 8
  %16 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, align 8
  %17 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, align 8
  %18 = alloca { ptr, ptr, i64, [3 x i64], [3 x i64] }, align 8
  %.idx = shl i64 %14, 3
  %19 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 %.idx)
  %20 = getelementptr inbounds nuw i8, ptr %1, i64 296
  %21 = load float, ptr %20, align 4, !tbaa !5
  %22 = getelementptr inbounds nuw i8, ptr %1, i64 292
  %23 = load float, ptr %22, align 4, !tbaa !5
  %24 = getelementptr inbounds nuw i8, ptr %1, i64 288
  %25 = load float, ptr %24, align 4, !tbaa !5
  %26 = getelementptr inbounds nuw i8, ptr %1, i64 200
  %27 = load float, ptr %26, align 4, !tbaa !5
  %28 = getelementptr inbounds nuw i8, ptr %1, i64 196
  %29 = load float, ptr %28, align 4, !tbaa !5
  %30 = getelementptr inbounds nuw i8, ptr %1, i64 192
  %31 = load float, ptr %30, align 4, !tbaa !5
  %32 = getelementptr inbounds nuw i8, ptr %1, i64 176
  %33 = load float, ptr %32, align 4, !tbaa !5
  %34 = getelementptr inbounds nuw i8, ptr %1, i64 172
  %35 = load float, ptr %34, align 4, !tbaa !5
  %36 = getelementptr inbounds nuw i8, ptr %1, i64 168
  %37 = load float, ptr %36, align 4, !tbaa !5
  %38 = getelementptr inbounds nuw i8, ptr %1, i64 92
  %39 = load float, ptr %38, align 4, !tbaa !5
  %40 = getelementptr inbounds nuw i8, ptr %1, i64 88
  %41 = load float, ptr %40, align 4, !tbaa !5
  %42 = getelementptr inbounds nuw i8, ptr %1, i64 84
  %43 = load float, ptr %42, align 4, !tbaa !5
  %44 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 96)
  %45 = ptrtoint ptr %44 to i64
  %46 = add i64 %45, 63
  %47 = and i64 %46, -64
  %48 = inttoptr i64 %47 to ptr
  store float 0x400921FB60000000, ptr %48, align 64, !tbaa !5
  %49 = getelementptr inbounds nuw i8, ptr %48, i64 4
  store float 0x400921FB60000000, ptr %49, align 4, !tbaa !5
  %50 = getelementptr inbounds nuw i8, ptr %48, i64 8
  store float 0x400921FB60000000, ptr %50, align 8, !tbaa !5
  %51 = getelementptr inbounds nuw i8, ptr %48, i64 12
  store float 0x400921FB60000000, ptr %51, align 4, !tbaa !5
  %52 = getelementptr inbounds nuw i8, ptr %48, i64 16
  store float 0x400921FB60000000, ptr %52, align 16, !tbaa !5
  %53 = getelementptr inbounds nuw i8, ptr %48, i64 20
  store float 0x400921FB60000000, ptr %53, align 4, !tbaa !5
  %54 = getelementptr inbounds nuw i8, ptr %48, i64 24
  store float 0x400921FB60000000, ptr %54, align 8, !tbaa !5
  %55 = getelementptr inbounds nuw i8, ptr %48, i64 28
  store float 0x400921FB60000000, ptr %55, align 4, !tbaa !5
  %56 = load float, ptr %10, align 4, !tbaa !5
  %57 = fmul float %56, 0x400921FB60000000
  store float %57, ptr %48, align 64, !tbaa !5
  %58 = getelementptr inbounds nuw i8, ptr %10, i64 4
  %59 = load float, ptr %58, align 4, !tbaa !5
  %60 = fmul float %59, 0x400921FB60000000
  store float %60, ptr %49, align 4, !tbaa !5
  %61 = getelementptr inbounds nuw i8, ptr %10, i64 8
  %62 = load float, ptr %61, align 4, !tbaa !5
  %63 = fmul float %62, 0x400921FB60000000
  store float %63, ptr %50, align 8, !tbaa !5
  %64 = getelementptr inbounds nuw i8, ptr %10, i64 12
  %65 = load float, ptr %64, align 4, !tbaa !5
  %66 = fmul float %65, 0x400921FB60000000
  store float %66, ptr %51, align 4, !tbaa !5
  %67 = getelementptr inbounds nuw i8, ptr %10, i64 16
  %68 = load float, ptr %67, align 4, !tbaa !5
  %69 = fmul float %68, 0x400921FB60000000
  store float %69, ptr %52, align 16, !tbaa !5
  %70 = getelementptr inbounds nuw i8, ptr %10, i64 20
  %71 = load float, ptr %70, align 4, !tbaa !5
  %72 = fmul float %71, 0x400921FB60000000
  store float %72, ptr %53, align 4, !tbaa !5
  %73 = getelementptr inbounds nuw i8, ptr %10, i64 24
  %74 = load float, ptr %73, align 4, !tbaa !5
  %75 = fmul float %74, 0x400921FB60000000
  store float %75, ptr %54, align 8, !tbaa !5
  %76 = getelementptr inbounds nuw i8, ptr %10, i64 28
  %77 = load float, ptr %76, align 4, !tbaa !5
  %78 = fmul float %77, 0x400921FB60000000
  store float %78, ptr %55, align 4, !tbaa !5
  %79 = fpext float %78 to double
  store double %79, ptr %19, align 8, !tbaa !7
  %80 = fpext float %43 to double
  %81 = getelementptr inbounds nuw i8, ptr %19, i64 8
  store double %80, ptr %81, align 8, !tbaa !7
  %82 = fpext float %41 to double
  %83 = getelementptr inbounds nuw i8, ptr %19, i64 16
  store double %82, ptr %83, align 8, !tbaa !7
  %84 = fpext float %39 to double
  %85 = getelementptr inbounds nuw i8, ptr %19, i64 24
  store double %84, ptr %85, align 8, !tbaa !7
  %86 = getelementptr inbounds nuw i8, ptr %1, i64 80
  %87 = load float, ptr %86, align 4, !tbaa !5
  %88 = getelementptr inbounds nuw i8, ptr %1, i64 76
  %89 = load float, ptr %88, align 4, !tbaa !5
  %90 = getelementptr inbounds nuw i8, ptr %1, i64 72
  %91 = load float, ptr %90, align 4, !tbaa !5
  %92 = fpext float %75 to double
  %93 = getelementptr inbounds nuw i8, ptr %19, i64 32
  store double %92, ptr %93, align 8, !tbaa !7
  %94 = fpext float %91 to double
  %95 = getelementptr inbounds nuw i8, ptr %19, i64 40
  store double %94, ptr %95, align 8, !tbaa !7
  %96 = fpext float %89 to double
  %97 = getelementptr inbounds nuw i8, ptr %19, i64 48
  store double %96, ptr %97, align 8, !tbaa !7
  %98 = fpext float %87 to double
  %99 = getelementptr inbounds nuw i8, ptr %19, i64 56
  store double %98, ptr %99, align 8, !tbaa !7
  %100 = getelementptr inbounds nuw i8, ptr %1, i64 68
  %101 = load float, ptr %100, align 4, !tbaa !5
  %102 = getelementptr inbounds nuw i8, ptr %1, i64 64
  %103 = load float, ptr %102, align 4, !tbaa !5
  %104 = getelementptr inbounds nuw i8, ptr %1, i64 60
  %105 = load float, ptr %104, align 4, !tbaa !5
  %106 = fpext float %72 to double
  %107 = getelementptr inbounds nuw i8, ptr %19, i64 64
  store double %106, ptr %107, align 8, !tbaa !7
  %108 = fpext float %105 to double
  %109 = getelementptr inbounds nuw i8, ptr %19, i64 72
  store double %108, ptr %109, align 8, !tbaa !7
  %110 = fpext float %103 to double
  %111 = getelementptr inbounds nuw i8, ptr %19, i64 80
  store double %110, ptr %111, align 8, !tbaa !7
  %112 = fpext float %101 to double
  %113 = getelementptr inbounds nuw i8, ptr %19, i64 88
  store double %112, ptr %113, align 8, !tbaa !7
  %114 = getelementptr inbounds nuw i8, ptr %1, i64 56
  %115 = load float, ptr %114, align 4, !tbaa !5
  %116 = getelementptr inbounds nuw i8, ptr %1, i64 52
  %117 = load float, ptr %116, align 4, !tbaa !5
  %118 = getelementptr inbounds nuw i8, ptr %1, i64 48
  %119 = load float, ptr %118, align 4, !tbaa !5
  %120 = fpext float %69 to double
  %121 = getelementptr inbounds nuw i8, ptr %19, i64 96
  store double %120, ptr %121, align 8, !tbaa !7
  %122 = fpext float %119 to double
  %123 = getelementptr inbounds nuw i8, ptr %19, i64 104
  store double %122, ptr %123, align 8, !tbaa !7
  %124 = fpext float %117 to double
  %125 = getelementptr inbounds nuw i8, ptr %19, i64 112
  store double %124, ptr %125, align 8, !tbaa !7
  %126 = fpext float %115 to double
  %127 = getelementptr inbounds nuw i8, ptr %19, i64 120
  store double %126, ptr %127, align 8, !tbaa !7
  %128 = getelementptr inbounds nuw i8, ptr %1, i64 44
  %129 = load float, ptr %128, align 4, !tbaa !5
  %130 = getelementptr inbounds nuw i8, ptr %1, i64 40
  %131 = load float, ptr %130, align 4, !tbaa !5
  %132 = getelementptr inbounds nuw i8, ptr %1, i64 36
  %133 = load float, ptr %132, align 4, !tbaa !5
  %134 = fpext float %66 to double
  %135 = getelementptr inbounds nuw i8, ptr %19, i64 128
  store double %134, ptr %135, align 8, !tbaa !7
  %136 = fpext float %133 to double
  %137 = getelementptr inbounds nuw i8, ptr %19, i64 136
  store double %136, ptr %137, align 8, !tbaa !7
  %138 = fpext float %131 to double
  %139 = getelementptr inbounds nuw i8, ptr %19, i64 144
  store double %138, ptr %139, align 8, !tbaa !7
  %140 = fpext float %129 to double
  %141 = getelementptr inbounds nuw i8, ptr %19, i64 152
  store double %140, ptr %141, align 8, !tbaa !7
  %142 = getelementptr inbounds nuw i8, ptr %1, i64 32
  %143 = load float, ptr %142, align 4, !tbaa !5
  %144 = getelementptr inbounds nuw i8, ptr %1, i64 28
  %145 = load float, ptr %144, align 4, !tbaa !5
  %146 = getelementptr inbounds nuw i8, ptr %1, i64 24
  %147 = load float, ptr %146, align 4, !tbaa !5
  %148 = fpext float %63 to double
  %149 = getelementptr inbounds nuw i8, ptr %19, i64 160
  store double %148, ptr %149, align 8, !tbaa !7
  %150 = fpext float %147 to double
  %151 = getelementptr inbounds nuw i8, ptr %19, i64 168
  store double %150, ptr %151, align 8, !tbaa !7
  %152 = fpext float %145 to double
  %153 = getelementptr inbounds nuw i8, ptr %19, i64 176
  store double %152, ptr %153, align 8, !tbaa !7
  %154 = fpext float %143 to double
  %155 = getelementptr inbounds nuw i8, ptr %19, i64 184
  store double %154, ptr %155, align 8, !tbaa !7
  %156 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %157 = load float, ptr %156, align 4, !tbaa !5
  %158 = getelementptr inbounds nuw i8, ptr %1, i64 4
  %159 = load float, ptr %158, align 4, !tbaa !5
  %160 = load float, ptr %1, align 4, !tbaa !5
  %161 = fpext float %57 to double
  %162 = getelementptr inbounds nuw i8, ptr %19, i64 192
  store double %161, ptr %162, align 8, !tbaa !7
  %163 = fpext float %160 to double
  %164 = getelementptr inbounds nuw i8, ptr %19, i64 200
  store double %163, ptr %164, align 8, !tbaa !7
  %165 = fpext float %159 to double
  %166 = getelementptr inbounds nuw i8, ptr %19, i64 208
  store double %165, ptr %166, align 8, !tbaa !7
  %167 = fpext float %157 to double
  %168 = getelementptr inbounds nuw i8, ptr %19, i64 216
  store double %167, ptr %168, align 8, !tbaa !7
  %169 = getelementptr inbounds nuw i8, ptr %1, i64 20
  %170 = load float, ptr %169, align 4, !tbaa !5
  %171 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %172 = load float, ptr %171, align 4, !tbaa !5
  %173 = getelementptr inbounds nuw i8, ptr %1, i64 12
  %174 = load float, ptr %173, align 4, !tbaa !5
  tail call void @_mlir_memref_to_llvm_free(ptr %44)
  %175 = fpext float %60 to double
  %176 = getelementptr inbounds nuw i8, ptr %19, i64 224
  store double %175, ptr %176, align 8, !tbaa !7
  %177 = fpext float %174 to double
  %178 = getelementptr inbounds nuw i8, ptr %19, i64 232
  store double %177, ptr %178, align 8, !tbaa !7
  %179 = fpext float %172 to double
  %180 = getelementptr inbounds nuw i8, ptr %19, i64 240
  store double %179, ptr %180, align 8, !tbaa !7
  %181 = fpext float %170 to double
  %182 = getelementptr inbounds nuw i8, ptr %19, i64 248
  store double %181, ptr %182, align 8, !tbaa !7
  %183 = fpext float %37 to double
  %184 = getelementptr inbounds nuw i8, ptr %19, i64 256
  store double %183, ptr %184, align 8, !tbaa !7
  %185 = fpext float %35 to double
  %186 = getelementptr inbounds nuw i8, ptr %19, i64 264
  store double %185, ptr %186, align 8, !tbaa !7
  %187 = fpext float %33 to double
  %188 = getelementptr inbounds nuw i8, ptr %19, i64 272
  store double %187, ptr %188, align 8, !tbaa !7
  %189 = getelementptr inbounds nuw i8, ptr %1, i64 152
  %190 = load float, ptr %189, align 4, !tbaa !5
  %191 = getelementptr inbounds nuw i8, ptr %1, i64 148
  %192 = load float, ptr %191, align 4, !tbaa !5
  %193 = getelementptr inbounds nuw i8, ptr %1, i64 144
  %194 = load float, ptr %193, align 4, !tbaa !5
  %195 = fpext float %194 to double
  %196 = getelementptr inbounds nuw i8, ptr %19, i64 280
  store double %195, ptr %196, align 8, !tbaa !7
  %197 = fpext float %192 to double
  %198 = getelementptr inbounds nuw i8, ptr %19, i64 288
  store double %197, ptr %198, align 8, !tbaa !7
  %199 = fpext float %190 to double
  %200 = getelementptr inbounds nuw i8, ptr %19, i64 296
  store double %199, ptr %200, align 8, !tbaa !7
  %201 = getelementptr inbounds nuw i8, ptr %1, i64 104
  %202 = load float, ptr %201, align 4, !tbaa !5
  %203 = getelementptr inbounds nuw i8, ptr %1, i64 100
  %204 = load float, ptr %203, align 4, !tbaa !5
  %205 = getelementptr inbounds nuw i8, ptr %1, i64 96
  %206 = load float, ptr %205, align 4, !tbaa !5
  %207 = fpext float %206 to double
  %208 = getelementptr inbounds nuw i8, ptr %19, i64 304
  store double %207, ptr %208, align 8, !tbaa !7
  %209 = fpext float %204 to double
  %210 = getelementptr inbounds nuw i8, ptr %19, i64 312
  store double %209, ptr %210, align 8, !tbaa !7
  %211 = fpext float %202 to double
  %212 = getelementptr inbounds nuw i8, ptr %19, i64 320
  store double %211, ptr %212, align 8, !tbaa !7
  %213 = getelementptr inbounds nuw i8, ptr %1, i64 128
  %214 = load float, ptr %213, align 4, !tbaa !5
  %215 = getelementptr inbounds nuw i8, ptr %1, i64 124
  %216 = load float, ptr %215, align 4, !tbaa !5
  %217 = getelementptr inbounds nuw i8, ptr %1, i64 120
  %218 = load float, ptr %217, align 4, !tbaa !5
  %219 = fpext float %218 to double
  %220 = getelementptr inbounds nuw i8, ptr %19, i64 328
  store double %219, ptr %220, align 8, !tbaa !7
  %221 = fpext float %216 to double
  %222 = getelementptr inbounds nuw i8, ptr %19, i64 336
  store double %221, ptr %222, align 8, !tbaa !7
  %223 = fpext float %214 to double
  %224 = getelementptr inbounds nuw i8, ptr %19, i64 344
  store double %223, ptr %224, align 8, !tbaa !7
  %225 = fpext float %31 to double
  %226 = getelementptr inbounds nuw i8, ptr %19, i64 352
  store double %225, ptr %226, align 8, !tbaa !7
  %227 = fpext float %29 to double
  %228 = getelementptr inbounds nuw i8, ptr %19, i64 360
  store double %227, ptr %228, align 8, !tbaa !7
  %229 = fpext float %27 to double
  %230 = getelementptr inbounds nuw i8, ptr %19, i64 368
  store double %229, ptr %230, align 8, !tbaa !7
  %231 = getelementptr inbounds nuw i8, ptr %1, i64 236
  %232 = load float, ptr %231, align 4, !tbaa !5
  %233 = getelementptr inbounds nuw i8, ptr %1, i64 232
  %234 = load float, ptr %233, align 4, !tbaa !5
  %235 = getelementptr inbounds nuw i8, ptr %1, i64 228
  %236 = load float, ptr %235, align 4, !tbaa !5
  %237 = getelementptr inbounds nuw i8, ptr %1, i64 164
  %238 = load float, ptr %237, align 4, !tbaa !5
  %239 = getelementptr inbounds nuw i8, ptr %1, i64 160
  %240 = load float, ptr %239, align 4, !tbaa !5
  %241 = getelementptr inbounds nuw i8, ptr %1, i64 156
  %242 = load float, ptr %241, align 4, !tbaa !5
  %243 = fpext float %242 to double
  %244 = getelementptr inbounds nuw i8, ptr %19, i64 376
  store double %243, ptr %244, align 8, !tbaa !7
  %245 = fpext float %240 to double
  %246 = getelementptr inbounds nuw i8, ptr %19, i64 384
  store double %245, ptr %246, align 8, !tbaa !7
  %247 = fpext float %238 to double
  %248 = getelementptr inbounds nuw i8, ptr %19, i64 392
  store double %247, ptr %248, align 8, !tbaa !7
  %249 = getelementptr inbounds nuw i8, ptr %1, i64 116
  %250 = load float, ptr %249, align 4, !tbaa !5
  %251 = getelementptr inbounds nuw i8, ptr %1, i64 112
  %252 = load float, ptr %251, align 4, !tbaa !5
  %253 = getelementptr inbounds nuw i8, ptr %1, i64 108
  %254 = load float, ptr %253, align 4, !tbaa !5
  %255 = fpext float %254 to double
  %256 = getelementptr inbounds nuw i8, ptr %19, i64 400
  store double %255, ptr %256, align 8, !tbaa !7
  %257 = fpext float %252 to double
  %258 = getelementptr inbounds nuw i8, ptr %19, i64 408
  store double %257, ptr %258, align 8, !tbaa !7
  %259 = fpext float %250 to double
  %260 = getelementptr inbounds nuw i8, ptr %19, i64 416
  store double %259, ptr %260, align 8, !tbaa !7
  %261 = getelementptr inbounds nuw i8, ptr %1, i64 140
  %262 = load float, ptr %261, align 4, !tbaa !5
  %263 = getelementptr inbounds nuw i8, ptr %1, i64 136
  %264 = load float, ptr %263, align 4, !tbaa !5
  %265 = getelementptr inbounds nuw i8, ptr %1, i64 132
  %266 = load float, ptr %265, align 4, !tbaa !5
  %267 = fpext float %266 to double
  %268 = getelementptr inbounds nuw i8, ptr %19, i64 424
  store double %267, ptr %268, align 8, !tbaa !7
  %269 = fpext float %264 to double
  %270 = getelementptr inbounds nuw i8, ptr %19, i64 432
  store double %269, ptr %270, align 8, !tbaa !7
  %271 = fpext float %262 to double
  %272 = getelementptr inbounds nuw i8, ptr %19, i64 440
  store double %271, ptr %272, align 8, !tbaa !7
  %273 = fpext float %236 to double
  %274 = getelementptr inbounds nuw i8, ptr %19, i64 448
  store double %273, ptr %274, align 8, !tbaa !7
  %275 = fpext float %234 to double
  %276 = getelementptr inbounds nuw i8, ptr %19, i64 456
  store double %275, ptr %276, align 8, !tbaa !7
  %277 = fpext float %232 to double
  %278 = getelementptr inbounds nuw i8, ptr %19, i64 464
  store double %277, ptr %278, align 8, !tbaa !7
  %279 = getelementptr inbounds nuw i8, ptr %1, i64 224
  %280 = load float, ptr %279, align 4, !tbaa !5
  %281 = getelementptr inbounds nuw i8, ptr %1, i64 220
  %282 = load float, ptr %281, align 4, !tbaa !5
  %283 = getelementptr inbounds nuw i8, ptr %1, i64 216
  %284 = load float, ptr %283, align 4, !tbaa !5
  %285 = fpext float %284 to double
  %286 = getelementptr inbounds nuw i8, ptr %19, i64 472
  store double %285, ptr %286, align 8, !tbaa !7
  %287 = fpext float %282 to double
  %288 = getelementptr inbounds nuw i8, ptr %19, i64 480
  store double %287, ptr %288, align 8, !tbaa !7
  %289 = fpext float %280 to double
  %290 = getelementptr inbounds nuw i8, ptr %19, i64 488
  store double %289, ptr %290, align 8, !tbaa !7
  %291 = getelementptr inbounds nuw i8, ptr %1, i64 260
  %292 = load float, ptr %291, align 4, !tbaa !5
  %293 = getelementptr inbounds nuw i8, ptr %1, i64 256
  %294 = load float, ptr %293, align 4, !tbaa !5
  %295 = getelementptr inbounds nuw i8, ptr %1, i64 252
  %296 = load float, ptr %295, align 4, !tbaa !5
  %297 = getelementptr inbounds nuw i8, ptr %1, i64 188
  %298 = load float, ptr %297, align 4, !tbaa !5
  %299 = getelementptr inbounds nuw i8, ptr %1, i64 184
  %300 = load float, ptr %299, align 4, !tbaa !5
  %301 = getelementptr inbounds nuw i8, ptr %1, i64 180
  %302 = load float, ptr %301, align 4, !tbaa !5
  %303 = fpext float %302 to double
  %304 = getelementptr inbounds nuw i8, ptr %19, i64 496
  store double %303, ptr %304, align 8, !tbaa !7
  %305 = fpext float %300 to double
  %306 = getelementptr inbounds nuw i8, ptr %19, i64 504
  store double %305, ptr %306, align 8, !tbaa !7
  %307 = fpext float %298 to double
  %308 = getelementptr inbounds nuw i8, ptr %19, i64 512
  store double %307, ptr %308, align 8, !tbaa !7
  %309 = fpext float %296 to double
  %310 = getelementptr inbounds nuw i8, ptr %19, i64 520
  store double %309, ptr %310, align 8, !tbaa !7
  %311 = fpext float %294 to double
  %312 = getelementptr inbounds nuw i8, ptr %19, i64 528
  store double %311, ptr %312, align 8, !tbaa !7
  %313 = fpext float %292 to double
  %314 = getelementptr inbounds nuw i8, ptr %19, i64 536
  store double %313, ptr %314, align 8, !tbaa !7
  %315 = fpext float %25 to double
  %316 = getelementptr inbounds nuw i8, ptr %19, i64 544
  store double %315, ptr %316, align 8, !tbaa !7
  %317 = fpext float %23 to double
  %318 = getelementptr inbounds nuw i8, ptr %19, i64 552
  store double %317, ptr %318, align 8, !tbaa !7
  %319 = fpext float %21 to double
  %320 = getelementptr inbounds nuw i8, ptr %19, i64 560
  store double %319, ptr %320, align 8, !tbaa !7
  %321 = getelementptr inbounds nuw i8, ptr %1, i64 344
  %322 = load float, ptr %321, align 4, !tbaa !5
  %323 = getelementptr inbounds nuw i8, ptr %1, i64 340
  %324 = load float, ptr %323, align 4, !tbaa !5
  %325 = getelementptr inbounds nuw i8, ptr %1, i64 336
  %326 = load float, ptr %325, align 4, !tbaa !5
  %327 = getelementptr inbounds nuw i8, ptr %1, i64 284
  %328 = load float, ptr %327, align 4, !tbaa !5
  %329 = getelementptr inbounds nuw i8, ptr %1, i64 280
  %330 = load float, ptr %329, align 4, !tbaa !5
  %331 = getelementptr inbounds nuw i8, ptr %1, i64 276
  %332 = load float, ptr %331, align 4, !tbaa !5
  %333 = fpext float %332 to double
  %334 = getelementptr inbounds nuw i8, ptr %19, i64 568
  store double %333, ptr %334, align 8, !tbaa !7
  %335 = fpext float %330 to double
  %336 = getelementptr inbounds nuw i8, ptr %19, i64 576
  store double %335, ptr %336, align 8, !tbaa !7
  %337 = fpext float %328 to double
  %338 = getelementptr inbounds nuw i8, ptr %19, i64 584
  store double %337, ptr %338, align 8, !tbaa !7
  %339 = getelementptr inbounds nuw i8, ptr %1, i64 212
  %340 = load float, ptr %339, align 4, !tbaa !5
  %341 = getelementptr inbounds nuw i8, ptr %1, i64 208
  %342 = load float, ptr %341, align 4, !tbaa !5
  %343 = getelementptr inbounds nuw i8, ptr %1, i64 204
  %344 = load float, ptr %343, align 4, !tbaa !5
  %345 = fpext float %344 to double
  %346 = getelementptr inbounds nuw i8, ptr %19, i64 592
  store double %345, ptr %346, align 8, !tbaa !7
  %347 = fpext float %342 to double
  %348 = getelementptr inbounds nuw i8, ptr %19, i64 600
  store double %347, ptr %348, align 8, !tbaa !7
  %349 = fpext float %340 to double
  %350 = getelementptr inbounds nuw i8, ptr %19, i64 608
  store double %349, ptr %350, align 8, !tbaa !7
  %351 = getelementptr inbounds nuw i8, ptr %1, i64 248
  %352 = load float, ptr %351, align 4, !tbaa !5
  %353 = getelementptr inbounds nuw i8, ptr %1, i64 244
  %354 = load float, ptr %353, align 4, !tbaa !5
  %355 = getelementptr inbounds nuw i8, ptr %1, i64 240
  %356 = load float, ptr %355, align 4, !tbaa !5
  %357 = fpext float %356 to double
  %358 = getelementptr inbounds nuw i8, ptr %19, i64 616
  store double %357, ptr %358, align 8, !tbaa !7
  %359 = fpext float %354 to double
  %360 = getelementptr inbounds nuw i8, ptr %19, i64 624
  store double %359, ptr %360, align 8, !tbaa !7
  %361 = fpext float %352 to double
  %362 = getelementptr inbounds nuw i8, ptr %19, i64 632
  store double %361, ptr %362, align 8, !tbaa !7
  %363 = fpext float %326 to double
  %364 = getelementptr inbounds nuw i8, ptr %19, i64 640
  store double %363, ptr %364, align 8, !tbaa !7
  %365 = fpext float %324 to double
  %366 = getelementptr inbounds nuw i8, ptr %19, i64 648
  store double %365, ptr %366, align 8, !tbaa !7
  %367 = fpext float %322 to double
  %368 = getelementptr inbounds nuw i8, ptr %19, i64 656
  store double %367, ptr %368, align 8, !tbaa !7
  %369 = getelementptr inbounds nuw i8, ptr %1, i64 308
  %370 = load float, ptr %369, align 4, !tbaa !5
  %371 = getelementptr inbounds nuw i8, ptr %1, i64 304
  %372 = load float, ptr %371, align 4, !tbaa !5
  %373 = getelementptr inbounds nuw i8, ptr %1, i64 300
  %374 = load float, ptr %373, align 4, !tbaa !5
  %375 = getelementptr inbounds nuw i8, ptr %1, i64 272
  %376 = load float, ptr %375, align 4, !tbaa !5
  %377 = getelementptr inbounds nuw i8, ptr %1, i64 268
  %378 = load float, ptr %377, align 4, !tbaa !5
  %379 = getelementptr inbounds nuw i8, ptr %1, i64 264
  %380 = load float, ptr %379, align 4, !tbaa !5
  %381 = fpext float %380 to double
  %382 = getelementptr inbounds nuw i8, ptr %19, i64 664
  store double %381, ptr %382, align 8, !tbaa !7
  %383 = fpext float %378 to double
  %384 = getelementptr inbounds nuw i8, ptr %19, i64 672
  store double %383, ptr %384, align 8, !tbaa !7
  %385 = fpext float %376 to double
  %386 = getelementptr inbounds nuw i8, ptr %19, i64 680
  store double %385, ptr %386, align 8, !tbaa !7
  %387 = fpext float %374 to double
  %388 = getelementptr inbounds nuw i8, ptr %19, i64 688
  store double %387, ptr %388, align 8, !tbaa !7
  %389 = fpext float %372 to double
  %390 = getelementptr inbounds nuw i8, ptr %19, i64 696
  store double %389, ptr %390, align 8, !tbaa !7
  %391 = fpext float %370 to double
  %392 = getelementptr inbounds nuw i8, ptr %19, i64 704
  store double %391, ptr %392, align 8, !tbaa !7
  %393 = getelementptr inbounds nuw i8, ptr %1, i64 356
  %394 = load float, ptr %393, align 4, !tbaa !5
  %395 = getelementptr inbounds nuw i8, ptr %1, i64 352
  %396 = load float, ptr %395, align 4, !tbaa !5
  %397 = getelementptr inbounds nuw i8, ptr %1, i64 348
  %398 = load float, ptr %397, align 4, !tbaa !5
  %399 = fpext float %398 to double
  %400 = getelementptr inbounds nuw i8, ptr %19, i64 712
  store double %399, ptr %400, align 8, !tbaa !7
  %401 = fpext float %396 to double
  %402 = getelementptr inbounds nuw i8, ptr %19, i64 720
  store double %401, ptr %402, align 8, !tbaa !7
  %403 = fpext float %394 to double
  %404 = getelementptr inbounds nuw i8, ptr %19, i64 728
  store double %403, ptr %404, align 8, !tbaa !7
  %405 = getelementptr inbounds nuw i8, ptr %1, i64 320
  %406 = load float, ptr %405, align 4, !tbaa !5
  %407 = getelementptr inbounds nuw i8, ptr %1, i64 316
  %408 = load float, ptr %407, align 4, !tbaa !5
  %409 = getelementptr inbounds nuw i8, ptr %1, i64 312
  %410 = load float, ptr %409, align 4, !tbaa !5
  %411 = fpext float %410 to double
  %412 = getelementptr inbounds nuw i8, ptr %19, i64 736
  store double %411, ptr %412, align 8, !tbaa !7
  %413 = fpext float %408 to double
  %414 = getelementptr inbounds nuw i8, ptr %19, i64 744
  store double %413, ptr %414, align 8, !tbaa !7
  %415 = fpext float %406 to double
  %416 = getelementptr inbounds nuw i8, ptr %19, i64 752
  store double %415, ptr %416, align 8, !tbaa !7
  %417 = getelementptr inbounds nuw i8, ptr %1, i64 368
  %418 = load float, ptr %417, align 4, !tbaa !5
  %419 = getelementptr inbounds nuw i8, ptr %1, i64 364
  %420 = load float, ptr %419, align 4, !tbaa !5
  %421 = getelementptr inbounds nuw i8, ptr %1, i64 360
  %422 = load float, ptr %421, align 4, !tbaa !5
  %423 = fpext float %422 to double
  %424 = getelementptr inbounds nuw i8, ptr %19, i64 760
  store double %423, ptr %424, align 8, !tbaa !7
  %425 = fpext float %420 to double
  %426 = getelementptr inbounds nuw i8, ptr %19, i64 768
  store double %425, ptr %426, align 8, !tbaa !7
  %427 = fpext float %418 to double
  %428 = getelementptr inbounds nuw i8, ptr %19, i64 776
  store double %427, ptr %428, align 8, !tbaa !7
  %429 = getelementptr inbounds nuw i8, ptr %1, i64 332
  %430 = load float, ptr %429, align 4, !tbaa !5
  %431 = getelementptr inbounds nuw i8, ptr %1, i64 328
  %432 = load float, ptr %431, align 4, !tbaa !5
  %433 = getelementptr inbounds nuw i8, ptr %1, i64 324
  %434 = load float, ptr %433, align 4, !tbaa !5
  %435 = fpext float %434 to double
  %436 = getelementptr inbounds nuw i8, ptr %19, i64 784
  store double %435, ptr %436, align 8, !tbaa !7
  %437 = fpext float %432 to double
  %438 = getelementptr inbounds nuw i8, ptr %19, i64 792
  store double %437, ptr %438, align 8, !tbaa !7
  %439 = fpext float %430 to double
  %440 = getelementptr inbounds nuw i8, ptr %19, i64 800
  store double %439, ptr %440, align 8, !tbaa !7
  %441 = getelementptr inbounds nuw i8, ptr %1, i64 380
  %442 = load float, ptr %441, align 4, !tbaa !5
  %443 = getelementptr inbounds nuw i8, ptr %1, i64 376
  %444 = load float, ptr %443, align 4, !tbaa !5
  %445 = getelementptr inbounds nuw i8, ptr %1, i64 372
  %446 = load float, ptr %445, align 4, !tbaa !5
  %447 = fpext float %446 to double
  %448 = getelementptr inbounds nuw i8, ptr %19, i64 808
  store double %447, ptr %448, align 8, !tbaa !7
  %449 = fpext float %444 to double
  %450 = getelementptr inbounds nuw i8, ptr %19, i64 816
  store double %449, ptr %450, align 8, !tbaa !7
  %451 = fpext float %442 to double
  %452 = getelementptr inbounds nuw i8, ptr %19, i64 824
  store double %451, ptr %452, align 8, !tbaa !7
  store ptr %0, ptr %18, align 8
  %.fca.1.gep = getelementptr inbounds nuw i8, ptr %18, i64 8
  store ptr %1, ptr %.fca.1.gep, align 8
  %.fca.2.gep = getelementptr inbounds nuw i8, ptr %18, i64 16
  store i64 %2, ptr %.fca.2.gep, align 8
  %.fca.3.0.gep = getelementptr inbounds nuw i8, ptr %18, i64 24
  store i64 %3, ptr %.fca.3.0.gep, align 8
  %.fca.3.1.gep = getelementptr inbounds nuw i8, ptr %18, i64 32
  store i64 %4, ptr %.fca.3.1.gep, align 8
  %.fca.3.2.gep = getelementptr inbounds nuw i8, ptr %18, i64 40
  store i64 %5, ptr %.fca.3.2.gep, align 8
  %.fca.4.0.gep = getelementptr inbounds nuw i8, ptr %18, i64 48
  store i64 %6, ptr %.fca.4.0.gep, align 8
  %.fca.4.1.gep = getelementptr inbounds nuw i8, ptr %18, i64 56
  store i64 %7, ptr %.fca.4.1.gep, align 8
  %.fca.4.2.gep = getelementptr inbounds nuw i8, ptr %18, i64 64
  store i64 %8, ptr %.fca.4.2.gep, align 8
  store ptr %9, ptr %17, align 8
  %.fca.1.gep107 = getelementptr inbounds nuw i8, ptr %17, i64 8
  store ptr %10, ptr %.fca.1.gep107, align 8
  %.fca.2.gep109 = getelementptr inbounds nuw i8, ptr %17, i64 16
  store i64 %11, ptr %.fca.2.gep109, align 8
  %.fca.3.0.gep111 = getelementptr inbounds nuw i8, ptr %17, i64 24
  store i64 %12, ptr %.fca.3.0.gep111, align 8
  %.fca.4.0.gep113 = getelementptr inbounds nuw i8, ptr %17, i64 32
  store i64 %13, ptr %.fca.4.0.gep113, align 8
  store ptr %19, ptr %16, align 8
  %.fca.1.gep117 = getelementptr inbounds nuw i8, ptr %16, i64 8
  store ptr %19, ptr %.fca.1.gep117, align 8
  %.fca.2.gep119 = getelementptr inbounds nuw i8, ptr %16, i64 16
  store i64 0, ptr %.fca.2.gep119, align 8
  %.fca.3.0.gep121 = getelementptr inbounds nuw i8, ptr %16, i64 24
  store i64 %14, ptr %.fca.3.0.gep121, align 8
  %.fca.4.0.gep123 = getelementptr inbounds nuw i8, ptr %16, i64 32
  store i64 1, ptr %.fca.4.0.gep123, align 8
  %453 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 8)
  %454 = insertvalue { ptr, ptr, i64 } poison, ptr %453, 0
  %455 = insertvalue { ptr, ptr, i64 } %454, ptr %453, 1
  %456 = insertvalue { ptr, ptr, i64 } %455, i64 0, 2
  store ptr %453, ptr %15, align 8
  %.fca.1.gep127 = getelementptr inbounds nuw i8, ptr %15, i64 8
  store ptr %453, ptr %.fca.1.gep127, align 8
  %.fca.2.gep129 = getelementptr inbounds nuw i8, ptr %15, i64 16
  store i64 0, ptr %.fca.2.gep129, align 8
  call void @qnode_forward_0.quantum(ptr nonnull %18, ptr nonnull %17, ptr nonnull %16, ptr nonnull %15)
  ret { ptr, ptr, i64 } %456
}

define void @setup() local_unnamed_addr {
  tail call void @__catalyst__rt__initialize(ptr null)
  ret void
}

define void @teardown() local_unnamed_addr {
  tail call void @__catalyst__rt__finalize()
  ret void
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #5

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fabs.f64(double) #6

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.exp.f64(double) #6

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.log.f64(double) #6

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #7

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #8

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #8

; Function Attrs: mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite)
declare noalias noundef ptr @malloc(i64 noundef) local_unnamed_addr #9

; Function Attrs: mustprogress nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite)
declare void @free(ptr allocptr noundef captures(none)) local_unnamed_addr #10

; Function Attrs: nofree nounwind willreturn allockind("alloc,zeroed") allocsize(0,1) memory(inaccessiblemem: readwrite)
declare noalias noundef ptr @calloc(i64 noundef, i64 noundef) local_unnamed_addr #11

attributes #0 = { "enzyme_math"="free" "prev_linkage"="0" }
attributes #1 = { "enzyme_allocator"="0" "enzyme_deallocator"="-1" "prev_linkage"="0" }
attributes #2 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) }
attributes #3 = { "prev_linkage"="0" }
attributes #4 = { noinline "prev_linkage"="0" }
attributes #5 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #6 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #7 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #8 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #9 = { mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite) "alloc-family"="malloc" }
attributes #10 = { mustprogress nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite) "alloc-family"="malloc" }
attributes #11 = { nofree nounwind willreturn allockind("alloc,zeroed") allocsize(0,1) memory(inaccessiblemem: readwrite) "alloc-family"="malloc" }
attributes #12 = { mustprogress willreturn }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{ptr @_mlir_memref_to_llvm_free}
!2 = !{!3, !3, i64 0}
!3 = !{!"int", !4, i64 0}
!4 = !{!"Catalyst TBAA"}
!5 = !{!6, !6, i64 0}
!6 = !{!"float", !4, i64 0}
!7 = !{!8, !8, i64 0}
!8 = !{!"double", !4, i64 0}
!9 = !{!10}
!10 = distinct !{!10, !11, !"primal"}
!11 = distinct !{!11, !" diff: %"}
!12 = !{!13}
!13 = distinct !{!13, !11, !"shadow_0"}
!14 = !{!15}
!15 = distinct !{!15, !16, !"primal"}
!16 = distinct !{!16, !" diff: %"}
!17 = !{!18}
!18 = distinct !{!18, !16, !"shadow_0"}
!19 = !{!20}
!20 = distinct !{!20, !21, !"primal"}
!21 = distinct !{!21, !" diff: %"}
!22 = !{!23}
!23 = distinct !{!23, !21, !"shadow_0"}
!24 = !{!25}
!25 = distinct !{!25, !26, !"primal"}
!26 = distinct !{!26, !" diff: %"}
!27 = !{!28}
!28 = distinct !{!28, !26, !"shadow_0"}
!29 = !{i64 8}
!30 = !{!31}
!31 = distinct !{!31, !32, !"primal"}
!32 = distinct !{!32, !" diff: %"}
!33 = !{!34}
!34 = distinct !{!34, !32, !"shadow_0"}
!35 = !{!36}
!36 = distinct !{!36, !37, !"primal"}
!37 = distinct !{!37, !" diff: %"}
!38 = !{!39}
!39 = distinct !{!39, !37, !"shadow_0"}
!40 = !{!41}
!41 = distinct !{!41, !42, !"primal"}
!42 = distinct !{!42, !" diff: %"}
!43 = !{!44}
!44 = distinct !{!44, !42, !"shadow_0"}
!45 = !{!46}
!46 = distinct !{!46, !47, !"primal"}
!47 = distinct !{!47, !" diff: %"}
!48 = !{!49}
!49 = distinct !{!49, !47, !"shadow_0"}
!50 = !{!51}
!51 = distinct !{!51, !52, !"primal"}
!52 = distinct !{!52, !" diff: %"}
!53 = !{!54}
!54 = distinct !{!54, !52, !"shadow_0"}
!55 = !{!56}
!56 = distinct !{!56, !57, !"shadow_0"}
!57 = distinct !{!57, !" diff: %"}
!58 = !{!59}
!59 = distinct !{!59, !57, !"primal"}
!60 = !{!61}
!61 = distinct !{!61, !62, !"shadow_0"}
!62 = distinct !{!62, !" diff: %"}
!63 = !{!64}
!64 = distinct !{!64, !62, !"primal"}
!65 = !{!66}
!66 = distinct !{!66, !67, !"primal"}
!67 = distinct !{!67, !" diff: %"}
!68 = !{!69}
!69 = distinct !{!69, !67, !"shadow_0"}
!70 = !{!71}
!71 = distinct !{!71, !72, !"shadow_0"}
!72 = distinct !{!72, !" diff: %"}
!73 = !{!74}
!74 = distinct !{!74, !72, !"primal"}
!75 = !{!76}
!76 = distinct !{!76, !77, !"shadow_0"}
!77 = distinct !{!77, !" diff: %"}
!78 = !{!79}
!79 = distinct !{!79, !77, !"primal"}
!80 = !{!81}
!81 = distinct !{!81, !82, !"shadow_0"}
!82 = distinct !{!82, !" diff: %"}
!83 = !{!84}
!84 = distinct !{!84, !82, !"primal"}
!85 = !{ptr @qnode_forward_0.quantum.augfwd}
!86 = !{ptr @qnode_forward_0.quantum.customqgrad}
