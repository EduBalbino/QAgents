; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@"{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}" = internal constant [54 x i8] c"{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}\00"
@LightningSimulator = internal constant [19 x i8] c"LightningSimulator\00"
@"/home/pichau/QAgents/.venv/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.so" = internal constant [107 x i8] c"/home/pichau/QAgents/.venv/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.so\00"
@enzyme_dupnoneed = linkonce constant i8 0
@enzyme_const = linkonce constant i8 0
@__enzyme_function_like_free = local_unnamed_addr global [2 x ptr] [ptr @_mlir_memref_to_llvm_free, ptr @freename]
@freename = linkonce constant [5 x i8] c"free\00"
@dealloc_indices = linkonce constant [3 x i8] c"-1\00"
@__enzyme_allocation_like = local_unnamed_addr global [4 x ptr] [ptr @_mlir_memref_to_llvm_alloc, ptr null, ptr @dealloc_indices, ptr @_mlir_memref_to_llvm_free]
@__enzyme_register_gradient_qnode_forward_0.quantum = local_unnamed_addr global [3 x ptr] [ptr @qnode_forward_0.quantum, ptr @qnode_forward_0.quantum.augfwd, ptr @qnode_forward_0.quantum.customqgrad]
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

declare void @__enzyme_autodiff0(...) local_unnamed_addr

declare void @_mlir_memref_to_llvm_free(ptr)

declare ptr @_mlir_memref_to_llvm_alloc(i64)

define { { ptr, ptr, i64, [3 x i64], [3 x i64] }, { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64, [1 x i64], [1 x i64] }, { ptr, ptr, i64, [1 x i64], [1 x i64] } } @jit_train_epoch_compiled(ptr readnone captures(none) %0, ptr readonly captures(none) %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, ptr readnone captures(none) %9, ptr readonly captures(none) %10, i64 %11, ptr readnone captures(none) %12, ptr readonly captures(none) %13, i64 %14, ptr %15, ptr %16, i64 %17, ptr readnone captures(none) %18, ptr readonly captures(none) %19, i64 %20, i64 %21, i64 %22, ptr %23, ptr %24, i64 %25, i64 %26, i64 %27, i64 %28, i64 %29, ptr %30, ptr %31, i64 %32, i64 %33, i64 %34, ptr %35, ptr %36, i64 %37, i64 %38, i64 %39) local_unnamed_addr {
.preheader395.preheader:
  %40 = load i32, ptr %19, align 4, !tbaa !1
  %41 = getelementptr inbounds nuw i8, ptr %19, i64 4
  %42 = load i32, ptr %41, align 4, !tbaa !1
  %43 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 80)
  %44 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 80)
  %45 = ptrtoint ptr %44 to i64
  %46 = add i64 %45, 63
  %47 = and i64 %46, -64
  %48 = inttoptr i64 %47 to ptr
  store i64 0, ptr %48, align 64, !tbaa !1
  %49 = getelementptr inbounds nuw i8, ptr %48, i64 8
  store i64 1, ptr %49, align 8, !tbaa !1
  %50 = ptrtoint ptr %43 to i64
  %51 = add i64 %50, 63
  %52 = and i64 %51, -64
  %53 = inttoptr i64 %52 to ptr
  store i64 1, ptr %53, align 64, !tbaa !1
  %54 = getelementptr inbounds nuw i8, ptr %53, i64 8
  store i64 1, ptr %54, align 8, !tbaa !1
  %55 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 80)
  %56 = ptrtoint ptr %55 to i64
  %57 = add i64 %56, 63
  %58 = and i64 %57, -64
  %59 = inttoptr i64 %58 to ptr
  %60 = load i64, ptr %53, align 64, !tbaa !1
  %61 = load i64, ptr %48, align 64, !tbaa !1
  %62 = mul i64 %61, %60
  store i64 %62, ptr %59, align 64, !tbaa !1
  %63 = load i64, ptr %54, align 8, !tbaa !1
  %64 = load i64, ptr %49, align 8, !tbaa !1
  %65 = mul i64 %64, %63
  %66 = getelementptr inbounds nuw i8, ptr %59, i64 8
  store i64 %65, ptr %66, align 8, !tbaa !1
  tail call void @_mlir_memref_to_llvm_free(ptr %44)
  store i64 32, ptr %53, align 64, !tbaa !1
  store i64 32, ptr %54, align 8, !tbaa !1
  %67 = load i64, ptr %59, align 64, !tbaa !1
  %68 = lshr i64 %67, 32
  store i64 %68, ptr %53, align 64, !tbaa !1
  %69 = load i64, ptr %66, align 8, !tbaa !1
  %70 = lshr i64 %69, 32
  store i64 %70, ptr %54, align 8, !tbaa !1
  %71 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %72 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %73 = ptrtoint ptr %72 to i64
  %74 = add i64 %73, 63
  %75 = and i64 %74, -64
  %76 = inttoptr i64 %75 to ptr
  %77 = load i64, ptr %59, align 64, !tbaa !1
  %78 = trunc i64 %77 to i32
  store i32 %78, ptr %76, align 64, !tbaa !1
  %79 = load i64, ptr %66, align 8, !tbaa !1
  %80 = trunc i64 %79 to i32
  %81 = getelementptr inbounds nuw i8, ptr %76, i64 4
  store i32 %80, ptr %81, align 4, !tbaa !1
  tail call void @_mlir_memref_to_llvm_free(ptr %55)
  %82 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %83 = ptrtoint ptr %82 to i64
  %84 = add i64 %83, 63
  %85 = and i64 %84, -64
  %86 = inttoptr i64 %85 to ptr
  %87 = load i64, ptr %53, align 64, !tbaa !1
  %88 = trunc i64 %87 to i32
  store i32 %88, ptr %86, align 64, !tbaa !1
  %89 = load i64, ptr %54, align 8, !tbaa !1
  %90 = trunc i64 %89 to i32
  %91 = getelementptr inbounds nuw i8, ptr %86, i64 4
  store i32 %90, ptr %91, align 4, !tbaa !1
  %92 = ptrtoint ptr %71 to i64
  %93 = add i64 %92, 63
  %94 = and i64 %93, -64
  %95 = inttoptr i64 %94 to ptr
  tail call void @_mlir_memref_to_llvm_free(ptr %43)
  %96 = load i32, ptr %19, align 4, !tbaa !1
  store i32 %96, ptr %95, align 64, !tbaa !1
  %97 = getelementptr inbounds nuw i8, ptr %95, i64 4
  store i32 %96, ptr %97, align 4, !tbaa !1
  %98 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %99 = ptrtoint ptr %98 to i64
  %100 = add i64 %99, 63
  %101 = and i64 %100, -64
  %102 = inttoptr i64 %101 to ptr
  %103 = load i32, ptr %86, align 64, !tbaa !1
  %104 = load i32, ptr %95, align 64, !tbaa !1
  %105 = add i32 %104, %103
  store i32 %105, ptr %102, align 64, !tbaa !1
  %106 = load i32, ptr %91, align 4, !tbaa !1
  %107 = load i32, ptr %97, align 4, !tbaa !1
  %108 = add i32 %107, %106
  %109 = getelementptr inbounds nuw i8, ptr %102, i64 4
  store i32 %108, ptr %109, align 4, !tbaa !1
  tail call void @_mlir_memref_to_llvm_free(ptr %82)
  %110 = load i32, ptr %41, align 4, !tbaa !1
  store i32 %110, ptr %95, align 64, !tbaa !1
  store i32 %110, ptr %97, align 4, !tbaa !1
  %111 = load i32, ptr %76, align 64, !tbaa !1
  %112 = add i32 %110, %111
  store i32 %112, ptr %95, align 64, !tbaa !1
  %113 = load i32, ptr %81, align 4, !tbaa !1
  %114 = add i32 %110, %113
  store i32 %114, ptr %97, align 4, !tbaa !1
  %115 = xor i32 %40, %42
  %116 = xor i32 %115, 466688986
  tail call void @_mlir_memref_to_llvm_free(ptr %72)
  %117 = load i32, ptr %41, align 4, !tbaa !1
  %118 = load i32, ptr %19, align 4, !tbaa !1
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
  store i32 %250, ptr %133, align 64, !tbaa !1
  store i32 %249, ptr %138, align 64, !tbaa !1
  %253 = load i32, ptr %.pn133408, align 4, !tbaa !1
  %254 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %255 = load i32, ptr %.pn113406, align 4, !tbaa !1
  %256 = load i32, ptr %.pn123407, align 4, !tbaa !1
  %257 = add i32 %256, %255
  store i32 %257, ptr %143, align 64, !tbaa !1
  %258 = getelementptr inbounds nuw i8, ptr %.pn113406, i64 4
  %259 = load i32, ptr %258, align 4, !tbaa !1
  %260 = getelementptr inbounds nuw i8, ptr %.pn123407, i64 4
  %261 = load i32, ptr %260, align 4, !tbaa !1
  %262 = add i32 %261, %259
  store i32 %262, ptr %235, align 4, !tbaa !1
  %263 = ptrtoint ptr %254 to i64
  %264 = add i64 %263, 63
  %265 = and i64 %264, -64
  %266 = inttoptr i64 %265 to ptr
  tail call void @_mlir_memref_to_llvm_free(ptr nonnull %.pn113406)
  %267 = load i32, ptr %.pn133408, align 4, !tbaa !1
  store i32 %267, ptr %266, align 64, !tbaa !1
  %268 = getelementptr inbounds nuw i8, ptr %266, i64 4
  store i32 %267, ptr %268, align 4, !tbaa !1
  %269 = load i32, ptr %.pn123407, align 4, !tbaa !1
  %270 = shl i32 %269, %267
  %271 = icmp ult i32 %267, 32
  %272 = select i1 %271, i32 %270, i32 0
  store i32 %272, ptr %148, align 64, !tbaa !1
  %273 = load i32, ptr %260, align 4, !tbaa !1
  %274 = load i32, ptr %268, align 4, !tbaa !1
  %275 = shl i32 %273, %274
  %276 = icmp ult i32 %274, 32
  %277 = select i1 %276, i32 %275, i32 0
  store i32 %277, ptr %236, align 4, !tbaa !1
  %278 = sub i32 32, %253
  store i32 %278, ptr %153, align 64, !tbaa !1
  store i32 %278, ptr %266, align 64, !tbaa !1
  %279 = load i32, ptr %153, align 64, !tbaa !1
  store i32 %279, ptr %268, align 4, !tbaa !1
  %280 = load i32, ptr %.pn123407, align 4, !tbaa !1
  %281 = lshr i32 %280, %278
  %282 = icmp ult i32 %278, 32
  %283 = select i1 %282, i32 %281, i32 0
  store i32 %283, ptr %266, align 64, !tbaa !1
  %284 = load i32, ptr %260, align 4, !tbaa !1
  %285 = lshr i32 %284, %279
  %286 = icmp ult i32 %279, 32
  %287 = select i1 %286, i32 %285, i32 0
  store i32 %287, ptr %268, align 4, !tbaa !1
  tail call void @_mlir_memref_to_llvm_free(ptr nonnull %.pn123407)
  %288 = load i32, ptr %148, align 64, !tbaa !1
  %289 = load i32, ptr %266, align 64, !tbaa !1
  %290 = or i32 %289, %288
  store i32 %290, ptr %266, align 64, !tbaa !1
  %291 = load i32, ptr %236, align 4, !tbaa !1
  %292 = load i32, ptr %268, align 4, !tbaa !1
  %293 = or i32 %292, %291
  store i32 %293, ptr %268, align 4, !tbaa !1
  %294 = load i32, ptr %143, align 64, !tbaa !1
  %295 = xor i32 %290, %294
  store i32 %295, ptr %158, align 64, !tbaa !1
  %296 = load i32, ptr %235, align 4, !tbaa !1
  %297 = load i32, ptr %268, align 4, !tbaa !1
  %298 = xor i32 %297, %296
  store i32 %298, ptr %237, align 4, !tbaa !1
  %299 = getelementptr inbounds nuw i8, ptr %.pn133408, i64 4
  %300 = load i32, ptr %299, align 4, !tbaa !1
  %301 = load i32, ptr %143, align 64, !tbaa !1
  %302 = add i32 %295, %301
  store i32 %302, ptr %163, align 64, !tbaa !1
  %303 = load i32, ptr %235, align 4, !tbaa !1
  %304 = load i32, ptr %237, align 4, !tbaa !1
  %305 = add i32 %304, %303
  store i32 %305, ptr %238, align 4, !tbaa !1
  %306 = load i32, ptr %299, align 4, !tbaa !1
  store i32 %306, ptr %266, align 64, !tbaa !1
  store i32 %306, ptr %268, align 4, !tbaa !1
  %307 = load i32, ptr %158, align 64, !tbaa !1
  %308 = shl i32 %307, %306
  %309 = icmp ult i32 %306, 32
  %310 = select i1 %309, i32 %308, i32 0
  store i32 %310, ptr %168, align 64, !tbaa !1
  %311 = load i32, ptr %237, align 4, !tbaa !1
  %312 = load i32, ptr %268, align 4, !tbaa !1
  %313 = shl i32 %311, %312
  %314 = icmp ult i32 %312, 32
  %315 = select i1 %314, i32 %313, i32 0
  store i32 %315, ptr %239, align 4, !tbaa !1
  %316 = sub i32 32, %300
  store i32 %316, ptr %173, align 64, !tbaa !1
  store i32 %316, ptr %266, align 64, !tbaa !1
  %317 = load i32, ptr %173, align 64, !tbaa !1
  store i32 %317, ptr %268, align 4, !tbaa !1
  %318 = load i32, ptr %158, align 64, !tbaa !1
  %319 = lshr i32 %318, %316
  %320 = icmp ult i32 %316, 32
  %321 = select i1 %320, i32 %319, i32 0
  store i32 %321, ptr %266, align 64, !tbaa !1
  %322 = load i32, ptr %237, align 4, !tbaa !1
  %323 = lshr i32 %322, %317
  %324 = icmp ult i32 %317, 32
  %325 = select i1 %324, i32 %323, i32 0
  store i32 %325, ptr %268, align 4, !tbaa !1
  %326 = load i32, ptr %168, align 64, !tbaa !1
  %327 = or i32 %321, %326
  store i32 %327, ptr %266, align 64, !tbaa !1
  %328 = load i32, ptr %239, align 4, !tbaa !1
  %329 = or i32 %325, %328
  store i32 %329, ptr %268, align 4, !tbaa !1
  %330 = load i32, ptr %163, align 64, !tbaa !1
  %331 = xor i32 %327, %330
  store i32 %331, ptr %178, align 64, !tbaa !1
  %332 = load i32, ptr %238, align 4, !tbaa !1
  %333 = load i32, ptr %268, align 4, !tbaa !1
  %334 = xor i32 %333, %332
  store i32 %334, ptr %240, align 4, !tbaa !1
  %335 = getelementptr inbounds nuw i8, ptr %.pn133408, i64 8
  %336 = load i32, ptr %335, align 4, !tbaa !1
  %337 = load i32, ptr %163, align 64, !tbaa !1
  %338 = add i32 %331, %337
  store i32 %338, ptr %183, align 64, !tbaa !1
  %339 = load i32, ptr %238, align 4, !tbaa !1
  %340 = load i32, ptr %240, align 4, !tbaa !1
  %341 = add i32 %340, %339
  store i32 %341, ptr %241, align 4, !tbaa !1
  %342 = load i32, ptr %335, align 4, !tbaa !1
  store i32 %342, ptr %266, align 64, !tbaa !1
  store i32 %342, ptr %268, align 4, !tbaa !1
  %343 = load i32, ptr %178, align 64, !tbaa !1
  %344 = shl i32 %343, %342
  %345 = icmp ult i32 %342, 32
  %346 = select i1 %345, i32 %344, i32 0
  store i32 %346, ptr %188, align 64, !tbaa !1
  %347 = load i32, ptr %240, align 4, !tbaa !1
  %348 = load i32, ptr %268, align 4, !tbaa !1
  %349 = shl i32 %347, %348
  %350 = icmp ult i32 %348, 32
  %351 = select i1 %350, i32 %349, i32 0
  store i32 %351, ptr %242, align 4, !tbaa !1
  %352 = sub i32 32, %336
  store i32 %352, ptr %193, align 64, !tbaa !1
  store i32 %352, ptr %266, align 64, !tbaa !1
  %353 = load i32, ptr %193, align 64, !tbaa !1
  store i32 %353, ptr %268, align 4, !tbaa !1
  %354 = load i32, ptr %178, align 64, !tbaa !1
  %355 = lshr i32 %354, %352
  %356 = icmp ult i32 %352, 32
  %357 = select i1 %356, i32 %355, i32 0
  store i32 %357, ptr %266, align 64, !tbaa !1
  %358 = load i32, ptr %240, align 4, !tbaa !1
  %359 = lshr i32 %358, %353
  %360 = icmp ult i32 %353, 32
  %361 = select i1 %360, i32 %359, i32 0
  store i32 %361, ptr %268, align 4, !tbaa !1
  %362 = load i32, ptr %188, align 64, !tbaa !1
  %363 = or i32 %357, %362
  store i32 %363, ptr %266, align 64, !tbaa !1
  %364 = load i32, ptr %242, align 4, !tbaa !1
  %365 = or i32 %361, %364
  store i32 %365, ptr %268, align 4, !tbaa !1
  %366 = load i32, ptr %183, align 64, !tbaa !1
  %367 = xor i32 %363, %366
  store i32 %367, ptr %198, align 64, !tbaa !1
  %368 = load i32, ptr %241, align 4, !tbaa !1
  %369 = load i32, ptr %268, align 4, !tbaa !1
  %370 = xor i32 %369, %368
  store i32 %370, ptr %243, align 4, !tbaa !1
  %371 = getelementptr inbounds nuw i8, ptr %.pn133408, i64 12
  %372 = load i32, ptr %371, align 4, !tbaa !1
  %373 = load i32, ptr %183, align 64, !tbaa !1
  %374 = add i32 %367, %373
  store i32 %374, ptr %203, align 64, !tbaa !1
  %375 = load i32, ptr %241, align 4, !tbaa !1
  %376 = load i32, ptr %243, align 4, !tbaa !1
  %377 = add i32 %376, %375
  store i32 %377, ptr %244, align 4, !tbaa !1
  %378 = load i32, ptr %371, align 4, !tbaa !1
  store i32 %378, ptr %266, align 64, !tbaa !1
  store i32 %378, ptr %268, align 4, !tbaa !1
  %379 = load i32, ptr %198, align 64, !tbaa !1
  %380 = shl i32 %379, %378
  %381 = icmp ult i32 %378, 32
  %382 = select i1 %381, i32 %380, i32 0
  store i32 %382, ptr %208, align 64, !tbaa !1
  %383 = load i32, ptr %243, align 4, !tbaa !1
  %384 = load i32, ptr %268, align 4, !tbaa !1
  %385 = shl i32 %383, %384
  %386 = icmp ult i32 %384, 32
  %387 = select i1 %386, i32 %385, i32 0
  store i32 %387, ptr %245, align 4, !tbaa !1
  %388 = sub i32 32, %372
  store i32 %388, ptr %213, align 64, !tbaa !1
  store i32 %388, ptr %266, align 64, !tbaa !1
  %389 = load i32, ptr %213, align 64, !tbaa !1
  store i32 %389, ptr %268, align 4, !tbaa !1
  %390 = load i32, ptr %198, align 64, !tbaa !1
  %391 = lshr i32 %390, %388
  %392 = icmp ult i32 %388, 32
  %393 = select i1 %392, i32 %391, i32 0
  store i32 %393, ptr %266, align 64, !tbaa !1
  %394 = load i32, ptr %243, align 4, !tbaa !1
  %395 = lshr i32 %394, %389
  %396 = icmp ult i32 %389, 32
  %397 = select i1 %396, i32 %395, i32 0
  store i32 %397, ptr %268, align 4, !tbaa !1
  %398 = load i32, ptr %208, align 64, !tbaa !1
  %399 = or i32 %393, %398
  store i32 %399, ptr %266, align 64, !tbaa !1
  %400 = load i32, ptr %245, align 4, !tbaa !1
  %401 = or i32 %397, %400
  store i32 %401, ptr %268, align 4, !tbaa !1
  %402 = load i32, ptr %203, align 64, !tbaa !1
  %403 = xor i32 %399, %402
  store i32 %403, ptr %218, align 64, !tbaa !1
  %404 = load i32, ptr %244, align 4, !tbaa !1
  %405 = load i32, ptr %268, align 4, !tbaa !1
  %406 = xor i32 %405, %404
  store i32 %406, ptr %246, align 4, !tbaa !1
  %407 = load i32, ptr %133, align 64, !tbaa !1
  store i32 %407, ptr %266, align 64, !tbaa !1
  store i32 %407, ptr %268, align 4, !tbaa !1
  %408 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %409 = ptrtoint ptr %408 to i64
  %410 = add i64 %409, 63
  %411 = and i64 %410, -64
  %412 = inttoptr i64 %411 to ptr
  %413 = load i32, ptr %203, align 64, !tbaa !1
  %414 = load i32, ptr %266, align 64, !tbaa !1
  %415 = add i32 %414, %413
  store i32 %415, ptr %412, align 64, !tbaa !1
  %416 = load i32, ptr %244, align 4, !tbaa !1
  %417 = load i32, ptr %268, align 4, !tbaa !1
  %418 = add i32 %417, %416
  %419 = getelementptr inbounds nuw i8, ptr %412, i64 4
  store i32 %418, ptr %419, align 4, !tbaa !1
  %420 = load i32, ptr %138, align 64, !tbaa !1
  store i32 %420, ptr %266, align 64, !tbaa !1
  store i32 %420, ptr %268, align 4, !tbaa !1
  %421 = load i32, ptr %218, align 64, !tbaa !1
  %422 = add i32 %420, %421
  store i32 %422, ptr %223, align 64, !tbaa !1
  %423 = load i32, ptr %246, align 4, !tbaa !1
  %424 = load i32, ptr %268, align 4, !tbaa !1
  %425 = add i32 %424, %423
  store i32 %425, ptr %247, align 4, !tbaa !1
  %426 = add nuw nsw i32 %251, 1
  store i32 %426, ptr %228, align 64, !tbaa !1
  store i32 %426, ptr %266, align 64, !tbaa !1
  %427 = load i32, ptr %228, align 64, !tbaa !1
  store i32 %427, ptr %268, align 4, !tbaa !1
  %428 = load i32, ptr %223, align 64, !tbaa !1
  %429 = add i32 %426, %428
  store i32 %429, ptr %266, align 64, !tbaa !1
  %430 = load i32, ptr %247, align 4, !tbaa !1
  %431 = add i32 %427, %430
  store i32 %431, ptr %268, align 4, !tbaa !1
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
  %454 = load i32, ptr %444, align 4, !tbaa !1
  store i32 %454, ptr %453, align 64, !tbaa !1
  %455 = load i32, ptr %442, align 4, !tbaa !1
  %456 = getelementptr i8, ptr %453, i64 4
  store i32 %455, ptr %456, align 4, !tbaa !1
  %457 = getelementptr i8, ptr %453, i64 8
  %.in325.1445 = getelementptr inbounds nuw i8, ptr %444, i64 4
  %458 = load i32, ptr %.in325.1445, align 4, !tbaa !1
  store i32 %458, ptr %457, align 8, !tbaa !1
  %.in325.1.1 = getelementptr inbounds nuw i8, ptr %442, i64 4
  %459 = load i32, ptr %.in325.1.1, align 4, !tbaa !1
  %460 = getelementptr i8, ptr %453, i64 12
  store i32 %459, ptr %460, align 4, !tbaa !1
  tail call void @_mlir_memref_to_llvm_free(ptr nonnull %444)
  tail call void @_mlir_memref_to_llvm_free(ptr nonnull %442)
  %461 = load i32, ptr %457, align 8, !tbaa !1
  %462 = getelementptr inbounds nuw i8, ptr %453, i64 12
  %463 = load i32, ptr %462, align 4, !tbaa !1
  %464 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 80)
  %465 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 80)
  %466 = ptrtoint ptr %465 to i64
  %467 = add i64 %466, 63
  %468 = and i64 %467, -64
  %469 = inttoptr i64 %468 to ptr
  store i64 0, ptr %469, align 64, !tbaa !1
  %470 = getelementptr inbounds nuw i8, ptr %469, i64 8
  store i64 1, ptr %470, align 8, !tbaa !1
  %471 = ptrtoint ptr %464 to i64
  %472 = add i64 %471, 63
  %473 = and i64 %472, -64
  %474 = inttoptr i64 %473 to ptr
  store i64 1, ptr %474, align 64, !tbaa !1
  %475 = getelementptr inbounds nuw i8, ptr %474, i64 8
  store i64 1, ptr %475, align 8, !tbaa !1
  %476 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 80)
  %477 = ptrtoint ptr %476 to i64
  %478 = add i64 %477, 63
  %479 = and i64 %478, -64
  %480 = inttoptr i64 %479 to ptr
  %481 = load i64, ptr %474, align 64, !tbaa !1
  %482 = load i64, ptr %469, align 64, !tbaa !1
  %483 = mul i64 %482, %481
  store i64 %483, ptr %480, align 64, !tbaa !1
  %484 = load i64, ptr %475, align 8, !tbaa !1
  %485 = load i64, ptr %470, align 8, !tbaa !1
  %486 = mul i64 %485, %484
  %487 = getelementptr inbounds nuw i8, ptr %480, i64 8
  store i64 %486, ptr %487, align 8, !tbaa !1
  tail call void @_mlir_memref_to_llvm_free(ptr %465)
  store i64 32, ptr %474, align 64, !tbaa !1
  store i64 32, ptr %475, align 8, !tbaa !1
  %488 = load i64, ptr %480, align 64, !tbaa !1
  %489 = lshr i64 %488, 32
  store i64 %489, ptr %474, align 64, !tbaa !1
  %490 = load i64, ptr %487, align 8, !tbaa !1
  %491 = lshr i64 %490, 32
  store i64 %491, ptr %475, align 8, !tbaa !1
  %492 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %493 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %494 = ptrtoint ptr %493 to i64
  %495 = add i64 %494, 63
  %496 = and i64 %495, -64
  %497 = inttoptr i64 %496 to ptr
  %498 = load i64, ptr %480, align 64, !tbaa !1
  %499 = trunc i64 %498 to i32
  store i32 %499, ptr %497, align 64, !tbaa !1
  %500 = load i64, ptr %487, align 8, !tbaa !1
  %501 = trunc i64 %500 to i32
  %502 = getelementptr inbounds nuw i8, ptr %497, i64 4
  store i32 %501, ptr %502, align 4, !tbaa !1
  tail call void @_mlir_memref_to_llvm_free(ptr %476)
  %503 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %504 = ptrtoint ptr %503 to i64
  %505 = add i64 %504, 63
  %506 = and i64 %505, -64
  %507 = inttoptr i64 %506 to ptr
  %508 = load i64, ptr %474, align 64, !tbaa !1
  %509 = trunc i64 %508 to i32
  store i32 %509, ptr %507, align 64, !tbaa !1
  %510 = load i64, ptr %475, align 8, !tbaa !1
  %511 = trunc i64 %510 to i32
  %512 = getelementptr inbounds nuw i8, ptr %507, i64 4
  store i32 %511, ptr %512, align 4, !tbaa !1
  %513 = ptrtoint ptr %492 to i64
  %514 = add i64 %513, 63
  %515 = and i64 %514, -64
  %516 = inttoptr i64 %515 to ptr
  tail call void @_mlir_memref_to_llvm_free(ptr %464)
  %517 = load i32, ptr %457, align 8, !tbaa !1
  store i32 %517, ptr %516, align 64, !tbaa !1
  %518 = getelementptr inbounds nuw i8, ptr %516, i64 4
  store i32 %517, ptr %518, align 4, !tbaa !1
  %519 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %520 = ptrtoint ptr %519 to i64
  %521 = add i64 %520, 63
  %522 = and i64 %521, -64
  %523 = inttoptr i64 %522 to ptr
  %524 = load i32, ptr %507, align 64, !tbaa !1
  %525 = load i32, ptr %516, align 64, !tbaa !1
  %526 = add i32 %525, %524
  store i32 %526, ptr %523, align 64, !tbaa !1
  %527 = load i32, ptr %512, align 4, !tbaa !1
  %528 = load i32, ptr %518, align 4, !tbaa !1
  %529 = add i32 %528, %527
  %530 = getelementptr inbounds nuw i8, ptr %523, i64 4
  store i32 %529, ptr %530, align 4, !tbaa !1
  tail call void @_mlir_memref_to_llvm_free(ptr %503)
  %531 = load i32, ptr %462, align 4, !tbaa !1
  store i32 %531, ptr %516, align 64, !tbaa !1
  store i32 %531, ptr %518, align 4, !tbaa !1
  %532 = load i32, ptr %497, align 64, !tbaa !1
  %533 = add i32 %531, %532
  store i32 %533, ptr %516, align 64, !tbaa !1
  %534 = load i32, ptr %502, align 4, !tbaa !1
  %535 = add i32 %531, %534
  store i32 %535, ptr %518, align 4, !tbaa !1
  %536 = xor i32 %461, %463
  %537 = xor i32 %536, 466688986
  tail call void @_mlir_memref_to_llvm_free(ptr %493)
  %538 = load i32, ptr %462, align 4, !tbaa !1
  %539 = load i32, ptr %457, align 8, !tbaa !1
  %540 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 80)
  %541 = ptrtoint ptr %540 to i64
  %542 = add i64 %541, 63
  %543 = and i64 %542, -64
  %544 = inttoptr i64 %543 to ptr
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 64 dereferenceable(16) %544, ptr noundef nonnull align 64 dereferenceable(16) @__constant_4xi32_3, i64 16, i1 false)
  %545 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 80)
  %546 = ptrtoint ptr %545 to i64
  %547 = add i64 %546, 63
  %548 = and i64 %547, -64
  %549 = inttoptr i64 %548 to ptr
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 64 dereferenceable(16) %549, ptr noundef nonnull align 64 dereferenceable(16) @__constant_4xi32, i64 16, i1 false)
  %550 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 68)
  %551 = ptrtoint ptr %550 to i64
  %552 = add i64 %551, 63
  %553 = and i64 %552, -64
  %554 = inttoptr i64 %553 to ptr
  %555 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 68)
  %556 = ptrtoint ptr %555 to i64
  %557 = add i64 %556, 63
  %558 = and i64 %557, -64
  %559 = inttoptr i64 %558 to ptr
  %560 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %561 = ptrtoint ptr %560 to i64
  %562 = add i64 %561, 63
  %563 = and i64 %562, -64
  %564 = inttoptr i64 %563 to ptr
  %565 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %566 = ptrtoint ptr %565 to i64
  %567 = add i64 %566, 63
  %568 = and i64 %567, -64
  %569 = inttoptr i64 %568 to ptr
  %570 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 68)
  %571 = ptrtoint ptr %570 to i64
  %572 = add i64 %571, 63
  %573 = and i64 %572, -64
  %574 = inttoptr i64 %573 to ptr
  %575 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %576 = ptrtoint ptr %575 to i64
  %577 = add i64 %576, 63
  %578 = and i64 %577, -64
  %579 = inttoptr i64 %578 to ptr
  %580 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %581 = ptrtoint ptr %580 to i64
  %582 = add i64 %581, 63
  %583 = and i64 %582, -64
  %584 = inttoptr i64 %583 to ptr
  %585 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %586 = ptrtoint ptr %585 to i64
  %587 = add i64 %586, 63
  %588 = and i64 %587, -64
  %589 = inttoptr i64 %588 to ptr
  %590 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 68)
  %591 = ptrtoint ptr %590 to i64
  %592 = add i64 %591, 63
  %593 = and i64 %592, -64
  %594 = inttoptr i64 %593 to ptr
  %595 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %596 = ptrtoint ptr %595 to i64
  %597 = add i64 %596, 63
  %598 = and i64 %597, -64
  %599 = inttoptr i64 %598 to ptr
  %600 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %601 = ptrtoint ptr %600 to i64
  %602 = add i64 %601, 63
  %603 = and i64 %602, -64
  %604 = inttoptr i64 %603 to ptr
  %605 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %606 = ptrtoint ptr %605 to i64
  %607 = add i64 %606, 63
  %608 = and i64 %607, -64
  %609 = inttoptr i64 %608 to ptr
  %610 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 68)
  %611 = ptrtoint ptr %610 to i64
  %612 = add i64 %611, 63
  %613 = and i64 %612, -64
  %614 = inttoptr i64 %613 to ptr
  %615 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %616 = ptrtoint ptr %615 to i64
  %617 = add i64 %616, 63
  %618 = and i64 %617, -64
  %619 = inttoptr i64 %618 to ptr
  %620 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %621 = ptrtoint ptr %620 to i64
  %622 = add i64 %621, 63
  %623 = and i64 %622, -64
  %624 = inttoptr i64 %623 to ptr
  %625 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %626 = ptrtoint ptr %625 to i64
  %627 = add i64 %626, 63
  %628 = and i64 %627, -64
  %629 = inttoptr i64 %628 to ptr
  %630 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 68)
  %631 = ptrtoint ptr %630 to i64
  %632 = add i64 %631, 63
  %633 = and i64 %632, -64
  %634 = inttoptr i64 %633 to ptr
  %635 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %636 = ptrtoint ptr %635 to i64
  %637 = add i64 %636, 63
  %638 = and i64 %637, -64
  %639 = inttoptr i64 %638 to ptr
  %640 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %641 = ptrtoint ptr %640 to i64
  %642 = add i64 %641, 63
  %643 = and i64 %642, -64
  %644 = inttoptr i64 %643 to ptr
  %645 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 68)
  %646 = ptrtoint ptr %645 to i64
  %647 = add i64 %646, 63
  %648 = and i64 %647, -64
  %649 = inttoptr i64 %648 to ptr
  %650 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 8)
  %651 = load i64, ptr %516, align 64
  store i64 %651, ptr %650, align 1
  tail call void @_mlir_memref_to_llvm_free(ptr %492)
  %652 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 8)
  %653 = load i64, ptr %523, align 64
  store i64 %653, ptr %652, align 1
  tail call void @_mlir_memref_to_llvm_free(ptr %519)
  %654 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 16)
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(16) %654, ptr noundef nonnull align 64 dereferenceable(16) %544, i64 16, i1 false)
  tail call void @_mlir_memref_to_llvm_free(ptr %540)
  %655 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 16)
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(16) %655, ptr noundef nonnull align 64 dereferenceable(16) %549, i64 16, i1 false)
  tail call void @_mlir_memref_to_llvm_free(ptr %545)
  %656 = getelementptr inbounds nuw i8, ptr %564, i64 4
  %657 = getelementptr inbounds nuw i8, ptr %569, i64 4
  %658 = getelementptr inbounds nuw i8, ptr %579, i64 4
  %659 = getelementptr inbounds nuw i8, ptr %584, i64 4
  %660 = getelementptr inbounds nuw i8, ptr %589, i64 4
  %661 = getelementptr inbounds nuw i8, ptr %599, i64 4
  %662 = getelementptr inbounds nuw i8, ptr %604, i64 4
  %663 = getelementptr inbounds nuw i8, ptr %609, i64 4
  %664 = getelementptr inbounds nuw i8, ptr %619, i64 4
  %665 = getelementptr inbounds nuw i8, ptr %624, i64 4
  %666 = getelementptr inbounds nuw i8, ptr %629, i64 4
  %667 = getelementptr inbounds nuw i8, ptr %639, i64 4
  %668 = getelementptr inbounds nuw i8, ptr %644, i64 4
  br label %.preheader366.preheader

.preheader366.preheader:                          ; preds = %.preheader370, %.preheader366.preheader
  %.pn183413 = phi ptr [ %655, %.preheader370 ], [ %868, %.preheader366.preheader ]
  %.pn173412 = phi ptr [ %654, %.preheader370 ], [ %867, %.preheader366.preheader ]
  %669 = phi i32 [ %539, %.preheader370 ], [ %671, %.preheader366.preheader ]
  %670 = phi i32 [ %537, %.preheader370 ], [ %669, %.preheader366.preheader ]
  %671 = phi i32 [ %538, %.preheader370 ], [ %670, %.preheader366.preheader ]
  %.pn163411 = phi ptr [ %650, %.preheader370 ], [ %863, %.preheader366.preheader ]
  %.pn153410 = phi ptr [ %652, %.preheader370 ], [ %865, %.preheader366.preheader ]
  %672 = phi i32 [ 0, %.preheader370 ], [ %847, %.preheader366.preheader ]
  %673 = phi i64 [ 0, %.preheader370 ], [ %869, %.preheader366.preheader ]
  store i32 %671, ptr %554, align 64, !tbaa !1
  store i32 %670, ptr %559, align 64, !tbaa !1
  %674 = load i32, ptr %.pn173412, align 4, !tbaa !1
  %675 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %676 = load i32, ptr %.pn153410, align 4, !tbaa !1
  %677 = load i32, ptr %.pn163411, align 4, !tbaa !1
  %678 = add i32 %677, %676
  store i32 %678, ptr %564, align 64, !tbaa !1
  %679 = getelementptr inbounds nuw i8, ptr %.pn153410, i64 4
  %680 = load i32, ptr %679, align 4, !tbaa !1
  %681 = getelementptr inbounds nuw i8, ptr %.pn163411, i64 4
  %682 = load i32, ptr %681, align 4, !tbaa !1
  %683 = add i32 %682, %680
  store i32 %683, ptr %656, align 4, !tbaa !1
  %684 = ptrtoint ptr %675 to i64
  %685 = add i64 %684, 63
  %686 = and i64 %685, -64
  %687 = inttoptr i64 %686 to ptr
  tail call void @_mlir_memref_to_llvm_free(ptr nonnull %.pn153410)
  %688 = load i32, ptr %.pn173412, align 4, !tbaa !1
  store i32 %688, ptr %687, align 64, !tbaa !1
  %689 = getelementptr inbounds nuw i8, ptr %687, i64 4
  store i32 %688, ptr %689, align 4, !tbaa !1
  %690 = load i32, ptr %.pn163411, align 4, !tbaa !1
  %691 = shl i32 %690, %688
  %692 = icmp ult i32 %688, 32
  %693 = select i1 %692, i32 %691, i32 0
  store i32 %693, ptr %569, align 64, !tbaa !1
  %694 = load i32, ptr %681, align 4, !tbaa !1
  %695 = load i32, ptr %689, align 4, !tbaa !1
  %696 = shl i32 %694, %695
  %697 = icmp ult i32 %695, 32
  %698 = select i1 %697, i32 %696, i32 0
  store i32 %698, ptr %657, align 4, !tbaa !1
  %699 = sub i32 32, %674
  store i32 %699, ptr %574, align 64, !tbaa !1
  store i32 %699, ptr %687, align 64, !tbaa !1
  %700 = load i32, ptr %574, align 64, !tbaa !1
  store i32 %700, ptr %689, align 4, !tbaa !1
  %701 = load i32, ptr %.pn163411, align 4, !tbaa !1
  %702 = lshr i32 %701, %699
  %703 = icmp ult i32 %699, 32
  %704 = select i1 %703, i32 %702, i32 0
  store i32 %704, ptr %687, align 64, !tbaa !1
  %705 = load i32, ptr %681, align 4, !tbaa !1
  %706 = lshr i32 %705, %700
  %707 = icmp ult i32 %700, 32
  %708 = select i1 %707, i32 %706, i32 0
  store i32 %708, ptr %689, align 4, !tbaa !1
  tail call void @_mlir_memref_to_llvm_free(ptr nonnull %.pn163411)
  %709 = load i32, ptr %569, align 64, !tbaa !1
  %710 = load i32, ptr %687, align 64, !tbaa !1
  %711 = or i32 %710, %709
  store i32 %711, ptr %687, align 64, !tbaa !1
  %712 = load i32, ptr %657, align 4, !tbaa !1
  %713 = load i32, ptr %689, align 4, !tbaa !1
  %714 = or i32 %713, %712
  store i32 %714, ptr %689, align 4, !tbaa !1
  %715 = load i32, ptr %564, align 64, !tbaa !1
  %716 = xor i32 %711, %715
  store i32 %716, ptr %579, align 64, !tbaa !1
  %717 = load i32, ptr %656, align 4, !tbaa !1
  %718 = load i32, ptr %689, align 4, !tbaa !1
  %719 = xor i32 %718, %717
  store i32 %719, ptr %658, align 4, !tbaa !1
  %720 = getelementptr inbounds nuw i8, ptr %.pn173412, i64 4
  %721 = load i32, ptr %720, align 4, !tbaa !1
  %722 = load i32, ptr %564, align 64, !tbaa !1
  %723 = add i32 %716, %722
  store i32 %723, ptr %584, align 64, !tbaa !1
  %724 = load i32, ptr %656, align 4, !tbaa !1
  %725 = load i32, ptr %658, align 4, !tbaa !1
  %726 = add i32 %725, %724
  store i32 %726, ptr %659, align 4, !tbaa !1
  %727 = load i32, ptr %720, align 4, !tbaa !1
  store i32 %727, ptr %687, align 64, !tbaa !1
  store i32 %727, ptr %689, align 4, !tbaa !1
  %728 = load i32, ptr %579, align 64, !tbaa !1
  %729 = shl i32 %728, %727
  %730 = icmp ult i32 %727, 32
  %731 = select i1 %730, i32 %729, i32 0
  store i32 %731, ptr %589, align 64, !tbaa !1
  %732 = load i32, ptr %658, align 4, !tbaa !1
  %733 = load i32, ptr %689, align 4, !tbaa !1
  %734 = shl i32 %732, %733
  %735 = icmp ult i32 %733, 32
  %736 = select i1 %735, i32 %734, i32 0
  store i32 %736, ptr %660, align 4, !tbaa !1
  %737 = sub i32 32, %721
  store i32 %737, ptr %594, align 64, !tbaa !1
  store i32 %737, ptr %687, align 64, !tbaa !1
  %738 = load i32, ptr %594, align 64, !tbaa !1
  store i32 %738, ptr %689, align 4, !tbaa !1
  %739 = load i32, ptr %579, align 64, !tbaa !1
  %740 = lshr i32 %739, %737
  %741 = icmp ult i32 %737, 32
  %742 = select i1 %741, i32 %740, i32 0
  store i32 %742, ptr %687, align 64, !tbaa !1
  %743 = load i32, ptr %658, align 4, !tbaa !1
  %744 = lshr i32 %743, %738
  %745 = icmp ult i32 %738, 32
  %746 = select i1 %745, i32 %744, i32 0
  store i32 %746, ptr %689, align 4, !tbaa !1
  %747 = load i32, ptr %589, align 64, !tbaa !1
  %748 = or i32 %742, %747
  store i32 %748, ptr %687, align 64, !tbaa !1
  %749 = load i32, ptr %660, align 4, !tbaa !1
  %750 = or i32 %746, %749
  store i32 %750, ptr %689, align 4, !tbaa !1
  %751 = load i32, ptr %584, align 64, !tbaa !1
  %752 = xor i32 %748, %751
  store i32 %752, ptr %599, align 64, !tbaa !1
  %753 = load i32, ptr %659, align 4, !tbaa !1
  %754 = load i32, ptr %689, align 4, !tbaa !1
  %755 = xor i32 %754, %753
  store i32 %755, ptr %661, align 4, !tbaa !1
  %756 = getelementptr inbounds nuw i8, ptr %.pn173412, i64 8
  %757 = load i32, ptr %756, align 4, !tbaa !1
  %758 = load i32, ptr %584, align 64, !tbaa !1
  %759 = add i32 %752, %758
  store i32 %759, ptr %604, align 64, !tbaa !1
  %760 = load i32, ptr %659, align 4, !tbaa !1
  %761 = load i32, ptr %661, align 4, !tbaa !1
  %762 = add i32 %761, %760
  store i32 %762, ptr %662, align 4, !tbaa !1
  %763 = load i32, ptr %756, align 4, !tbaa !1
  store i32 %763, ptr %687, align 64, !tbaa !1
  store i32 %763, ptr %689, align 4, !tbaa !1
  %764 = load i32, ptr %599, align 64, !tbaa !1
  %765 = shl i32 %764, %763
  %766 = icmp ult i32 %763, 32
  %767 = select i1 %766, i32 %765, i32 0
  store i32 %767, ptr %609, align 64, !tbaa !1
  %768 = load i32, ptr %661, align 4, !tbaa !1
  %769 = load i32, ptr %689, align 4, !tbaa !1
  %770 = shl i32 %768, %769
  %771 = icmp ult i32 %769, 32
  %772 = select i1 %771, i32 %770, i32 0
  store i32 %772, ptr %663, align 4, !tbaa !1
  %773 = sub i32 32, %757
  store i32 %773, ptr %614, align 64, !tbaa !1
  store i32 %773, ptr %687, align 64, !tbaa !1
  %774 = load i32, ptr %614, align 64, !tbaa !1
  store i32 %774, ptr %689, align 4, !tbaa !1
  %775 = load i32, ptr %599, align 64, !tbaa !1
  %776 = lshr i32 %775, %773
  %777 = icmp ult i32 %773, 32
  %778 = select i1 %777, i32 %776, i32 0
  store i32 %778, ptr %687, align 64, !tbaa !1
  %779 = load i32, ptr %661, align 4, !tbaa !1
  %780 = lshr i32 %779, %774
  %781 = icmp ult i32 %774, 32
  %782 = select i1 %781, i32 %780, i32 0
  store i32 %782, ptr %689, align 4, !tbaa !1
  %783 = load i32, ptr %609, align 64, !tbaa !1
  %784 = or i32 %778, %783
  store i32 %784, ptr %687, align 64, !tbaa !1
  %785 = load i32, ptr %663, align 4, !tbaa !1
  %786 = or i32 %782, %785
  store i32 %786, ptr %689, align 4, !tbaa !1
  %787 = load i32, ptr %604, align 64, !tbaa !1
  %788 = xor i32 %784, %787
  store i32 %788, ptr %619, align 64, !tbaa !1
  %789 = load i32, ptr %662, align 4, !tbaa !1
  %790 = load i32, ptr %689, align 4, !tbaa !1
  %791 = xor i32 %790, %789
  store i32 %791, ptr %664, align 4, !tbaa !1
  %792 = getelementptr inbounds nuw i8, ptr %.pn173412, i64 12
  %793 = load i32, ptr %792, align 4, !tbaa !1
  %794 = load i32, ptr %604, align 64, !tbaa !1
  %795 = add i32 %788, %794
  store i32 %795, ptr %624, align 64, !tbaa !1
  %796 = load i32, ptr %662, align 4, !tbaa !1
  %797 = load i32, ptr %664, align 4, !tbaa !1
  %798 = add i32 %797, %796
  store i32 %798, ptr %665, align 4, !tbaa !1
  %799 = load i32, ptr %792, align 4, !tbaa !1
  store i32 %799, ptr %687, align 64, !tbaa !1
  store i32 %799, ptr %689, align 4, !tbaa !1
  %800 = load i32, ptr %619, align 64, !tbaa !1
  %801 = shl i32 %800, %799
  %802 = icmp ult i32 %799, 32
  %803 = select i1 %802, i32 %801, i32 0
  store i32 %803, ptr %629, align 64, !tbaa !1
  %804 = load i32, ptr %664, align 4, !tbaa !1
  %805 = load i32, ptr %689, align 4, !tbaa !1
  %806 = shl i32 %804, %805
  %807 = icmp ult i32 %805, 32
  %808 = select i1 %807, i32 %806, i32 0
  store i32 %808, ptr %666, align 4, !tbaa !1
  %809 = sub i32 32, %793
  store i32 %809, ptr %634, align 64, !tbaa !1
  store i32 %809, ptr %687, align 64, !tbaa !1
  %810 = load i32, ptr %634, align 64, !tbaa !1
  store i32 %810, ptr %689, align 4, !tbaa !1
  %811 = load i32, ptr %619, align 64, !tbaa !1
  %812 = lshr i32 %811, %809
  %813 = icmp ult i32 %809, 32
  %814 = select i1 %813, i32 %812, i32 0
  store i32 %814, ptr %687, align 64, !tbaa !1
  %815 = load i32, ptr %664, align 4, !tbaa !1
  %816 = lshr i32 %815, %810
  %817 = icmp ult i32 %810, 32
  %818 = select i1 %817, i32 %816, i32 0
  store i32 %818, ptr %689, align 4, !tbaa !1
  %819 = load i32, ptr %629, align 64, !tbaa !1
  %820 = or i32 %814, %819
  store i32 %820, ptr %687, align 64, !tbaa !1
  %821 = load i32, ptr %666, align 4, !tbaa !1
  %822 = or i32 %818, %821
  store i32 %822, ptr %689, align 4, !tbaa !1
  %823 = load i32, ptr %624, align 64, !tbaa !1
  %824 = xor i32 %820, %823
  store i32 %824, ptr %639, align 64, !tbaa !1
  %825 = load i32, ptr %665, align 4, !tbaa !1
  %826 = load i32, ptr %689, align 4, !tbaa !1
  %827 = xor i32 %826, %825
  store i32 %827, ptr %667, align 4, !tbaa !1
  %828 = load i32, ptr %554, align 64, !tbaa !1
  store i32 %828, ptr %687, align 64, !tbaa !1
  store i32 %828, ptr %689, align 4, !tbaa !1
  %829 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %830 = ptrtoint ptr %829 to i64
  %831 = add i64 %830, 63
  %832 = and i64 %831, -64
  %833 = inttoptr i64 %832 to ptr
  %834 = load i32, ptr %624, align 64, !tbaa !1
  %835 = load i32, ptr %687, align 64, !tbaa !1
  %836 = add i32 %835, %834
  store i32 %836, ptr %833, align 64, !tbaa !1
  %837 = load i32, ptr %665, align 4, !tbaa !1
  %838 = load i32, ptr %689, align 4, !tbaa !1
  %839 = add i32 %838, %837
  %840 = getelementptr inbounds nuw i8, ptr %833, i64 4
  store i32 %839, ptr %840, align 4, !tbaa !1
  %841 = load i32, ptr %559, align 64, !tbaa !1
  store i32 %841, ptr %687, align 64, !tbaa !1
  store i32 %841, ptr %689, align 4, !tbaa !1
  %842 = load i32, ptr %639, align 64, !tbaa !1
  %843 = add i32 %841, %842
  store i32 %843, ptr %644, align 64, !tbaa !1
  %844 = load i32, ptr %667, align 4, !tbaa !1
  %845 = load i32, ptr %689, align 4, !tbaa !1
  %846 = add i32 %845, %844
  store i32 %846, ptr %668, align 4, !tbaa !1
  %847 = add nuw nsw i32 %672, 1
  store i32 %847, ptr %649, align 64, !tbaa !1
  store i32 %847, ptr %687, align 64, !tbaa !1
  %848 = load i32, ptr %649, align 64, !tbaa !1
  store i32 %848, ptr %689, align 4, !tbaa !1
  %849 = load i32, ptr %644, align 64, !tbaa !1
  %850 = add i32 %847, %849
  store i32 %850, ptr %687, align 64, !tbaa !1
  %851 = load i32, ptr %668, align 4, !tbaa !1
  %852 = add i32 %848, %851
  store i32 %852, ptr %689, align 4, !tbaa !1
  %853 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 80)
  %854 = ptrtoint ptr %853 to i64
  %855 = add i64 %854, 63
  %856 = and i64 %855, -64
  %857 = inttoptr i64 %856 to ptr
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 64 dereferenceable(16) %857, ptr noundef nonnull align 1 dereferenceable(16) %.pn183413, i64 16, i1 false)
  tail call void @_mlir_memref_to_llvm_free(ptr nonnull %.pn183413)
  %858 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 80)
  %859 = ptrtoint ptr %858 to i64
  %860 = add i64 %859, 63
  %861 = and i64 %860, -64
  %862 = inttoptr i64 %861 to ptr
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 64 dereferenceable(16) %862, ptr noundef nonnull align 1 dereferenceable(16) %.pn173412, i64 16, i1 false)
  tail call void @_mlir_memref_to_llvm_free(ptr nonnull %.pn173412)
  %863 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 8)
  %864 = load i64, ptr %687, align 64
  store i64 %864, ptr %863, align 1
  tail call void @_mlir_memref_to_llvm_free(ptr %675)
  %865 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 8)
  %866 = load i64, ptr %833, align 64
  store i64 %866, ptr %865, align 1
  tail call void @_mlir_memref_to_llvm_free(ptr %829)
  %867 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 16)
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(16) %867, ptr noundef nonnull align 64 dereferenceable(16) %857, i64 16, i1 false)
  tail call void @_mlir_memref_to_llvm_free(ptr %853)
  %868 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 16)
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(16) %868, ptr noundef nonnull align 64 dereferenceable(16) %862, i64 16, i1 false)
  tail call void @_mlir_memref_to_llvm_free(ptr %858)
  %869 = add nuw nsw i64 %673, 1
  %exitcond446.not = icmp eq i64 %869, 5
  br i1 %exitcond446.not, label %.preheader344, label %.preheader366.preheader

.preheader344:                                    ; preds = %.preheader366.preheader
  tail call void @_mlir_memref_to_llvm_free(ptr nonnull %868)
  tail call void @_mlir_memref_to_llvm_free(ptr nonnull %867)
  tail call void @_mlir_memref_to_llvm_free(ptr %645)
  tail call void @_mlir_memref_to_llvm_free(ptr %640)
  tail call void @_mlir_memref_to_llvm_free(ptr %635)
  tail call void @_mlir_memref_to_llvm_free(ptr %630)
  tail call void @_mlir_memref_to_llvm_free(ptr %625)
  tail call void @_mlir_memref_to_llvm_free(ptr %620)
  tail call void @_mlir_memref_to_llvm_free(ptr %615)
  tail call void @_mlir_memref_to_llvm_free(ptr %610)
  tail call void @_mlir_memref_to_llvm_free(ptr %605)
  tail call void @_mlir_memref_to_llvm_free(ptr %600)
  tail call void @_mlir_memref_to_llvm_free(ptr %595)
  tail call void @_mlir_memref_to_llvm_free(ptr %590)
  tail call void @_mlir_memref_to_llvm_free(ptr %585)
  tail call void @_mlir_memref_to_llvm_free(ptr %580)
  tail call void @_mlir_memref_to_llvm_free(ptr %575)
  tail call void @_mlir_memref_to_llvm_free(ptr %570)
  tail call void @_mlir_memref_to_llvm_free(ptr %565)
  tail call void @_mlir_memref_to_llvm_free(ptr %560)
  tail call void @_mlir_memref_to_llvm_free(ptr %555)
  tail call void @_mlir_memref_to_llvm_free(ptr %550)
  %870 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 80)
  %871 = ptrtoint ptr %870 to i64
  %872 = add i64 %871, 63
  %873 = and i64 %872, -64
  %874 = inttoptr i64 %873 to ptr
  %875 = load i32, ptr %865, align 4, !tbaa !1
  store i32 %875, ptr %874, align 64, !tbaa !1
  %876 = load i32, ptr %863, align 4, !tbaa !1
  %877 = getelementptr i8, ptr %874, i64 4
  store i32 %876, ptr %877, align 4, !tbaa !1
  %878 = getelementptr i8, ptr %874, i64 8
  %.in323.1448 = getelementptr inbounds nuw i8, ptr %865, i64 4
  %879 = load i32, ptr %.in323.1448, align 4, !tbaa !1
  store i32 %879, ptr %878, align 8, !tbaa !1
  %.in323.1.1 = getelementptr inbounds nuw i8, ptr %863, i64 4
  %880 = load i32, ptr %.in323.1.1, align 4, !tbaa !1
  %881 = getelementptr i8, ptr %874, i64 12
  store i32 %880, ptr %881, align 4, !tbaa !1
  tail call void @_mlir_memref_to_llvm_free(ptr nonnull %865)
  tail call void @_mlir_memref_to_llvm_free(ptr nonnull %863)
  %882 = load i32, ptr %874, align 64, !tbaa !1
  %883 = load i32, ptr %877, align 4, !tbaa !1
  %884 = xor i32 %882, %883
  %885 = xor i32 %884, 466688986
  %886 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 80)
  %887 = ptrtoint ptr %886 to i64
  %888 = add i64 %887, 63
  %889 = and i64 %888, -64
  %890 = inttoptr i64 %889 to ptr
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 64 dereferenceable(16) %890, ptr noundef nonnull align 64 dereferenceable(16) @__constant_4xi32_3, i64 16, i1 false)
  %891 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 80)
  %892 = ptrtoint ptr %891 to i64
  %893 = add i64 %892, 63
  %894 = and i64 %893, -64
  %895 = inttoptr i64 %894 to ptr
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 64 dereferenceable(16) %895, ptr noundef nonnull align 64 dereferenceable(16) @__constant_4xi32, i64 16, i1 false)
  %896 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 16)
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(16) %896, ptr noundef nonnull align 64 dereferenceable(16) %890, i64 16, i1 false)
  tail call void @_mlir_memref_to_llvm_free(ptr %886)
  %897 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 16)
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(16) %897, ptr noundef nonnull align 64 dereferenceable(16) %895, i64 16, i1 false)
  tail call void @_mlir_memref_to_llvm_free(ptr %891)
  br label %898

898:                                              ; preds = %.preheader344, %898
  %.pn203415 = phi ptr [ %897, %.preheader344 ], [ %968, %898 ]
  %.pn193414 = phi ptr [ %896, %.preheader344 ], [ %967, %898 ]
  %899 = phi i32 [ %882, %.preheader344 ], [ %901, %898 ]
  %900 = phi i32 [ %885, %.preheader344 ], [ %899, %898 ]
  %901 = phi i32 [ %883, %.preheader344 ], [ %900, %898 ]
  %902 = phi i32 [ %883, %.preheader344 ], [ %956, %898 ]
  %903 = phi i32 [ %882, %.preheader344 ], [ %954, %898 ]
  %904 = phi i32 [ 0, %.preheader344 ], [ %906, %898 ]
  %905 = phi i64 [ 0, %.preheader344 ], [ %969, %898 ]
  %906 = add nuw nsw i32 %904, 1
  %907 = load i32, ptr %.pn193414, align 4, !tbaa !1
  %908 = add i32 %902, %903
  %909 = shl i32 %902, %907
  %910 = icmp ult i32 %907, 32
  %911 = select i1 %910, i32 %909, i32 0
  %912 = sub i32 32, %907
  %913 = lshr i32 %902, %912
  %914 = icmp ult i32 %912, 32
  %915 = select i1 %914, i32 %913, i32 0
  %916 = or i32 %915, %911
  %917 = xor i32 %916, %908
  %918 = getelementptr inbounds nuw i8, ptr %.pn193414, i64 4
  %919 = load i32, ptr %918, align 4, !tbaa !1
  %920 = add i32 %917, %908
  %921 = shl i32 %917, %919
  %922 = icmp ult i32 %919, 32
  %923 = select i1 %922, i32 %921, i32 0
  %924 = sub i32 32, %919
  %925 = lshr i32 %917, %924
  %926 = icmp ult i32 %924, 32
  %927 = select i1 %926, i32 %925, i32 0
  %928 = or i32 %923, %927
  %929 = xor i32 %928, %920
  %930 = getelementptr inbounds nuw i8, ptr %.pn193414, i64 8
  %931 = load i32, ptr %930, align 4, !tbaa !1
  %932 = add i32 %929, %920
  %933 = shl i32 %929, %931
  %934 = icmp ult i32 %931, 32
  %935 = select i1 %934, i32 %933, i32 0
  %936 = sub i32 32, %931
  %937 = lshr i32 %929, %936
  %938 = icmp ult i32 %936, 32
  %939 = select i1 %938, i32 %937, i32 0
  %940 = or i32 %935, %939
  %941 = xor i32 %940, %932
  %942 = getelementptr inbounds nuw i8, ptr %.pn193414, i64 12
  %943 = load i32, ptr %942, align 4, !tbaa !1
  %944 = add i32 %941, %932
  %945 = shl i32 %941, %943
  %946 = icmp ult i32 %943, 32
  %947 = select i1 %946, i32 %945, i32 0
  %948 = sub i32 32, %943
  %949 = lshr i32 %941, %948
  %950 = icmp ult i32 %948, 32
  %951 = select i1 %950, i32 %949, i32 0
  %952 = or i32 %947, %951
  %953 = xor i32 %952, %944
  %954 = add i32 %944, %901
  %955 = add i32 %900, %906
  %956 = add i32 %955, %953
  %957 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 80)
  %958 = ptrtoint ptr %957 to i64
  %959 = add i64 %958, 63
  %960 = and i64 %959, -64
  %961 = inttoptr i64 %960 to ptr
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 64 dereferenceable(16) %961, ptr noundef nonnull align 1 dereferenceable(16) %.pn203415, i64 16, i1 false)
  tail call void @_mlir_memref_to_llvm_free(ptr nonnull %.pn203415)
  %962 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 80)
  %963 = ptrtoint ptr %962 to i64
  %964 = add i64 %963, 63
  %965 = and i64 %964, -64
  %966 = inttoptr i64 %965 to ptr
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 64 dereferenceable(16) %966, ptr noundef nonnull align 1 dereferenceable(16) %.pn193414, i64 16, i1 false)
  tail call void @_mlir_memref_to_llvm_free(ptr nonnull %.pn193414)
  %967 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 16)
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(16) %967, ptr noundef nonnull align 64 dereferenceable(16) %961, i64 16, i1 false)
  tail call void @_mlir_memref_to_llvm_free(ptr %957)
  %968 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 16)
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(16) %968, ptr noundef nonnull align 64 dereferenceable(16) %966, i64 16, i1 false)
  tail call void @_mlir_memref_to_llvm_free(ptr %962)
  %969 = add nuw nsw i64 %905, 1
  %exitcond449.not = icmp eq i64 %969, 5
  br i1 %exitcond449.not, label %970, label %898

970:                                              ; preds = %898
  tail call void @_mlir_memref_to_llvm_free(ptr nonnull %968)
  tail call void @_mlir_memref_to_llvm_free(ptr nonnull %967)
  %971 = load i32, ptr %878, align 8, !tbaa !1
  %972 = getelementptr inbounds nuw i8, ptr %874, i64 12
  %973 = load i32, ptr %972, align 4, !tbaa !1
  %974 = xor i32 %971, %973
  %975 = xor i32 %974, 466688986
  tail call void @_mlir_memref_to_llvm_free(ptr %870)
  %976 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 80)
  %977 = ptrtoint ptr %976 to i64
  %978 = add i64 %977, 63
  %979 = and i64 %978, -64
  %980 = inttoptr i64 %979 to ptr
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 64 dereferenceable(16) %980, ptr noundef nonnull align 64 dereferenceable(16) @__constant_4xi32_3, i64 16, i1 false)
  %981 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 80)
  %982 = ptrtoint ptr %981 to i64
  %983 = add i64 %982, 63
  %984 = and i64 %983, -64
  %985 = inttoptr i64 %984 to ptr
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 64 dereferenceable(16) %985, ptr noundef nonnull align 64 dereferenceable(16) @__constant_4xi32, i64 16, i1 false)
  %986 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 16)
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(16) %986, ptr noundef nonnull align 64 dereferenceable(16) %980, i64 16, i1 false)
  tail call void @_mlir_memref_to_llvm_free(ptr %976)
  %987 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 16)
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(16) %987, ptr noundef nonnull align 64 dereferenceable(16) %985, i64 16, i1 false)
  tail call void @_mlir_memref_to_llvm_free(ptr %981)
  br label %988

988:                                              ; preds = %970, %988
  %.pn223417 = phi ptr [ %987, %970 ], [ %1058, %988 ]
  %.pn213416 = phi ptr [ %986, %970 ], [ %1057, %988 ]
  %989 = phi i32 [ %971, %970 ], [ %991, %988 ]
  %990 = phi i32 [ %975, %970 ], [ %989, %988 ]
  %991 = phi i32 [ %973, %970 ], [ %990, %988 ]
  %992 = phi i32 [ %973, %970 ], [ %1046, %988 ]
  %993 = phi i32 [ %971, %970 ], [ %1044, %988 ]
  %994 = phi i32 [ 0, %970 ], [ %996, %988 ]
  %995 = phi i64 [ 0, %970 ], [ %1059, %988 ]
  %996 = add nuw nsw i32 %994, 1
  %997 = load i32, ptr %.pn213416, align 4, !tbaa !1
  %998 = add i32 %992, %993
  %999 = shl i32 %992, %997
  %1000 = icmp ult i32 %997, 32
  %1001 = select i1 %1000, i32 %999, i32 0
  %1002 = sub i32 32, %997
  %1003 = lshr i32 %992, %1002
  %1004 = icmp ult i32 %1002, 32
  %1005 = select i1 %1004, i32 %1003, i32 0
  %1006 = or i32 %1005, %1001
  %1007 = xor i32 %1006, %998
  %1008 = getelementptr inbounds nuw i8, ptr %.pn213416, i64 4
  %1009 = load i32, ptr %1008, align 4, !tbaa !1
  %1010 = add i32 %1007, %998
  %1011 = shl i32 %1007, %1009
  %1012 = icmp ult i32 %1009, 32
  %1013 = select i1 %1012, i32 %1011, i32 0
  %1014 = sub i32 32, %1009
  %1015 = lshr i32 %1007, %1014
  %1016 = icmp ult i32 %1014, 32
  %1017 = select i1 %1016, i32 %1015, i32 0
  %1018 = or i32 %1013, %1017
  %1019 = xor i32 %1018, %1010
  %1020 = getelementptr inbounds nuw i8, ptr %.pn213416, i64 8
  %1021 = load i32, ptr %1020, align 4, !tbaa !1
  %1022 = add i32 %1019, %1010
  %1023 = shl i32 %1019, %1021
  %1024 = icmp ult i32 %1021, 32
  %1025 = select i1 %1024, i32 %1023, i32 0
  %1026 = sub i32 32, %1021
  %1027 = lshr i32 %1019, %1026
  %1028 = icmp ult i32 %1026, 32
  %1029 = select i1 %1028, i32 %1027, i32 0
  %1030 = or i32 %1025, %1029
  %1031 = xor i32 %1030, %1022
  %1032 = getelementptr inbounds nuw i8, ptr %.pn213416, i64 12
  %1033 = load i32, ptr %1032, align 4, !tbaa !1
  %1034 = add i32 %1031, %1022
  %1035 = shl i32 %1031, %1033
  %1036 = icmp ult i32 %1033, 32
  %1037 = select i1 %1036, i32 %1035, i32 0
  %1038 = sub i32 32, %1033
  %1039 = lshr i32 %1031, %1038
  %1040 = icmp ult i32 %1038, 32
  %1041 = select i1 %1040, i32 %1039, i32 0
  %1042 = or i32 %1037, %1041
  %1043 = xor i32 %1042, %1034
  %1044 = add i32 %1034, %991
  %1045 = add i32 %990, %996
  %1046 = add i32 %1045, %1043
  %1047 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 80)
  %1048 = ptrtoint ptr %1047 to i64
  %1049 = add i64 %1048, 63
  %1050 = and i64 %1049, -64
  %1051 = inttoptr i64 %1050 to ptr
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 64 dereferenceable(16) %1051, ptr noundef nonnull align 1 dereferenceable(16) %.pn223417, i64 16, i1 false)
  tail call void @_mlir_memref_to_llvm_free(ptr nonnull %.pn223417)
  %1052 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 80)
  %1053 = ptrtoint ptr %1052 to i64
  %1054 = add i64 %1053, 63
  %1055 = and i64 %1054, -64
  %1056 = inttoptr i64 %1055 to ptr
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 64 dereferenceable(16) %1056, ptr noundef nonnull align 1 dereferenceable(16) %.pn213416, i64 16, i1 false)
  tail call void @_mlir_memref_to_llvm_free(ptr nonnull %.pn213416)
  %1057 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 16)
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(16) %1057, ptr noundef nonnull align 64 dereferenceable(16) %1051, i64 16, i1 false)
  tail call void @_mlir_memref_to_llvm_free(ptr %1047)
  %1058 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 16)
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(16) %1058, ptr noundef nonnull align 64 dereferenceable(16) %1056, i64 16, i1 false)
  tail call void @_mlir_memref_to_llvm_free(ptr %1052)
  %1059 = add nuw nsw i64 %995, 1
  %exitcond450.not = icmp eq i64 %1059, 5
  br i1 %exitcond450.not, label %1060, label %988

1060:                                             ; preds = %988
  %1061 = zext i32 %954 to i64
  %1062 = zext i32 %956 to i64
  %1063 = shl nuw i64 %1061, 32
  %1064 = or disjoint i64 %1063, %1062
  tail call void @_mlir_memref_to_llvm_free(ptr nonnull %1058)
  tail call void @_mlir_memref_to_llvm_free(ptr nonnull %1057)
  %1065 = zext i32 %1044 to i64
  %1066 = zext i32 %1046 to i64
  %1067 = shl nuw i64 %1065, 32
  %1068 = or disjoint i64 %1067, %1066
  %1069 = urem i64 %1064, 96
  %1070 = shl nuw nsw i64 %1069, 6
  %1071 = urem i64 %1068, 96
  %1072 = add nuw nsw i64 %1071, %1070
  %.lhs.trunc = trunc nuw nsw i64 %1072 to i16
  %1073 = urem i16 %.lhs.trunc, 96
  %.zext = zext nneg i16 %1073 to i64
  %1074 = load float, ptr %10, align 4, !tbaa !4
  %1075 = load float, ptr %13, align 4, !tbaa !4
  %1076 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 68)
  %1077 = ptrtoint ptr %1076 to i64
  %1078 = add i64 %1077, 63
  %1079 = and i64 %1078, -64
  %1080 = inttoptr i64 %1079 to ptr
  %1081 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 68)
  %1082 = ptrtoint ptr %1081 to i64
  %1083 = add i64 %1082, 63
  %1084 = and i64 %1083, -64
  %1085 = inttoptr i64 %1084 to ptr
  %1086 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 448)
  %1087 = ptrtoint ptr %1086 to i64
  %1088 = add i64 %1087, 63
  %1089 = and i64 %1088, -64
  %1090 = inttoptr i64 %1089 to ptr
  %1091 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 96)
  %1092 = ptrtoint ptr %1091 to i64
  %1093 = add i64 %1092, 63
  %1094 = and i64 %1093, -64
  %1095 = inttoptr i64 %1094 to ptr
  %1096 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %1097 = ptrtoint ptr %1096 to i64
  %1098 = add i64 %1097, 63
  %1099 = and i64 %1098, -64
  %1100 = inttoptr i64 %1099 to ptr
  %1101 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 384)
  %1102 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 4)
  %1103 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 4)
  %1104 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 8)
  %1105 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 448)
  %1106 = ptrtoint ptr %1105 to i64
  %1107 = add i64 %1106, 63
  %1108 = and i64 %1107, -64
  %1109 = inttoptr i64 %1108 to ptr
  %1110 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 384)
  %1111 = shl i64 %3, 2
  %1112 = mul i64 %1111, %4
  %1113 = mul i64 %1112, %5
  %1114 = getelementptr inbounds float, ptr %1, i64 %2
  tail call void @llvm.memcpy.p0.p0.i64(ptr align 1 %1110, ptr align 1 %1114, i64 %1113, i1 false)
  %1115 = getelementptr inbounds nuw i8, ptr %1095, i64 4
  %1116 = getelementptr inbounds nuw i8, ptr %1095, i64 8
  %1117 = getelementptr inbounds nuw i8, ptr %1095, i64 12
  %1118 = getelementptr inbounds nuw i8, ptr %1095, i64 16
  %1119 = getelementptr inbounds nuw i8, ptr %1095, i64 20
  %1120 = getelementptr inbounds nuw i8, ptr %1095, i64 24
  %1121 = getelementptr inbounds nuw i8, ptr %1095, i64 28
  %1122 = getelementptr inbounds nuw i8, ptr %1101, i64 4
  %1123 = getelementptr inbounds nuw i8, ptr %1101, i64 8
  %1124 = getelementptr inbounds nuw i8, ptr %1101, i64 12
  %1125 = getelementptr inbounds nuw i8, ptr %1101, i64 16
  %1126 = getelementptr inbounds nuw i8, ptr %1101, i64 20
  %1127 = getelementptr inbounds nuw i8, ptr %1101, i64 24
  %1128 = getelementptr inbounds nuw i8, ptr %1101, i64 28
  %1129 = getelementptr inbounds nuw i8, ptr %1101, i64 32
  %1130 = getelementptr inbounds nuw i8, ptr %1101, i64 36
  %1131 = getelementptr inbounds nuw i8, ptr %1101, i64 40
  %1132 = getelementptr inbounds nuw i8, ptr %1101, i64 44
  %1133 = getelementptr inbounds nuw i8, ptr %1101, i64 48
  %1134 = getelementptr inbounds nuw i8, ptr %1101, i64 52
  %1135 = getelementptr inbounds nuw i8, ptr %1101, i64 56
  %1136 = getelementptr inbounds nuw i8, ptr %1101, i64 60
  %1137 = getelementptr inbounds nuw i8, ptr %1101, i64 64
  %1138 = getelementptr inbounds nuw i8, ptr %1101, i64 68
  %1139 = getelementptr inbounds nuw i8, ptr %1101, i64 72
  %1140 = getelementptr inbounds nuw i8, ptr %1101, i64 76
  %1141 = getelementptr inbounds nuw i8, ptr %1101, i64 80
  %1142 = getelementptr inbounds nuw i8, ptr %1101, i64 84
  %1143 = getelementptr inbounds nuw i8, ptr %1101, i64 88
  %1144 = getelementptr inbounds nuw i8, ptr %1101, i64 92
  %1145 = getelementptr inbounds nuw i8, ptr %1101, i64 96
  %1146 = getelementptr inbounds nuw i8, ptr %1101, i64 100
  %1147 = getelementptr inbounds nuw i8, ptr %1101, i64 104
  %1148 = getelementptr inbounds nuw i8, ptr %1101, i64 108
  %1149 = getelementptr inbounds nuw i8, ptr %1101, i64 112
  %1150 = getelementptr inbounds nuw i8, ptr %1101, i64 116
  %1151 = getelementptr inbounds nuw i8, ptr %1101, i64 120
  %1152 = getelementptr inbounds nuw i8, ptr %1101, i64 124
  %1153 = getelementptr inbounds nuw i8, ptr %1101, i64 128
  %1154 = getelementptr inbounds nuw i8, ptr %1101, i64 132
  %1155 = getelementptr inbounds nuw i8, ptr %1101, i64 136
  %1156 = getelementptr inbounds nuw i8, ptr %1101, i64 140
  %1157 = getelementptr inbounds nuw i8, ptr %1101, i64 144
  %1158 = getelementptr inbounds nuw i8, ptr %1101, i64 148
  %1159 = getelementptr inbounds nuw i8, ptr %1101, i64 152
  %1160 = getelementptr inbounds nuw i8, ptr %1101, i64 156
  %1161 = getelementptr inbounds nuw i8, ptr %1101, i64 160
  %1162 = getelementptr inbounds nuw i8, ptr %1101, i64 164
  %1163 = getelementptr inbounds nuw i8, ptr %1101, i64 168
  %1164 = getelementptr inbounds nuw i8, ptr %1101, i64 172
  %1165 = getelementptr inbounds nuw i8, ptr %1101, i64 176
  %1166 = getelementptr inbounds nuw i8, ptr %1101, i64 180
  %1167 = getelementptr inbounds nuw i8, ptr %1101, i64 184
  %1168 = getelementptr inbounds nuw i8, ptr %1101, i64 188
  %1169 = getelementptr inbounds nuw i8, ptr %1101, i64 192
  %1170 = getelementptr inbounds nuw i8, ptr %1101, i64 196
  %1171 = getelementptr inbounds nuw i8, ptr %1101, i64 200
  %1172 = getelementptr inbounds nuw i8, ptr %1101, i64 204
  %1173 = getelementptr inbounds nuw i8, ptr %1101, i64 208
  %1174 = getelementptr inbounds nuw i8, ptr %1101, i64 212
  %1175 = getelementptr inbounds nuw i8, ptr %1101, i64 216
  %1176 = getelementptr inbounds nuw i8, ptr %1101, i64 220
  %1177 = getelementptr inbounds nuw i8, ptr %1101, i64 224
  %1178 = getelementptr inbounds nuw i8, ptr %1101, i64 228
  %1179 = getelementptr inbounds nuw i8, ptr %1101, i64 232
  %1180 = getelementptr inbounds nuw i8, ptr %1101, i64 236
  %1181 = getelementptr inbounds nuw i8, ptr %1101, i64 240
  %1182 = getelementptr inbounds nuw i8, ptr %1101, i64 244
  %1183 = getelementptr inbounds nuw i8, ptr %1101, i64 248
  %1184 = getelementptr inbounds nuw i8, ptr %1101, i64 252
  %1185 = getelementptr inbounds nuw i8, ptr %1101, i64 256
  %1186 = getelementptr inbounds nuw i8, ptr %1101, i64 260
  %1187 = getelementptr inbounds nuw i8, ptr %1101, i64 264
  %1188 = getelementptr inbounds nuw i8, ptr %1101, i64 268
  %1189 = getelementptr inbounds nuw i8, ptr %1101, i64 272
  %1190 = getelementptr inbounds nuw i8, ptr %1101, i64 276
  %1191 = getelementptr inbounds nuw i8, ptr %1101, i64 280
  %1192 = getelementptr inbounds nuw i8, ptr %1101, i64 284
  %1193 = getelementptr inbounds nuw i8, ptr %1101, i64 288
  %1194 = getelementptr inbounds nuw i8, ptr %1101, i64 292
  %1195 = getelementptr inbounds nuw i8, ptr %1101, i64 296
  %1196 = getelementptr inbounds nuw i8, ptr %1101, i64 300
  %1197 = getelementptr inbounds nuw i8, ptr %1101, i64 304
  %1198 = getelementptr inbounds nuw i8, ptr %1101, i64 308
  %1199 = getelementptr inbounds nuw i8, ptr %1101, i64 312
  %1200 = getelementptr inbounds nuw i8, ptr %1101, i64 316
  %1201 = getelementptr inbounds nuw i8, ptr %1101, i64 320
  %1202 = getelementptr inbounds nuw i8, ptr %1101, i64 324
  %1203 = getelementptr inbounds nuw i8, ptr %1101, i64 328
  %1204 = getelementptr inbounds nuw i8, ptr %1101, i64 332
  %1205 = getelementptr inbounds nuw i8, ptr %1101, i64 336
  %1206 = getelementptr inbounds nuw i8, ptr %1101, i64 340
  %1207 = getelementptr inbounds nuw i8, ptr %1101, i64 344
  %1208 = getelementptr inbounds nuw i8, ptr %1101, i64 348
  %1209 = getelementptr inbounds nuw i8, ptr %1101, i64 352
  %1210 = getelementptr inbounds nuw i8, ptr %1101, i64 356
  %1211 = getelementptr inbounds nuw i8, ptr %1101, i64 360
  %1212 = getelementptr inbounds nuw i8, ptr %1101, i64 364
  %1213 = getelementptr inbounds nuw i8, ptr %1101, i64 368
  %1214 = getelementptr inbounds nuw i8, ptr %1101, i64 372
  %1215 = getelementptr inbounds nuw i8, ptr %1101, i64 376
  %1216 = getelementptr inbounds nuw i8, ptr %1101, i64 380
  %1217 = getelementptr inbounds nuw i8, ptr %1109, i64 4
  %1218 = getelementptr inbounds nuw i8, ptr %1109, i64 8
  %1219 = getelementptr inbounds nuw i8, ptr %1109, i64 12
  %1220 = getelementptr inbounds nuw i8, ptr %1109, i64 16
  %1221 = getelementptr inbounds nuw i8, ptr %1109, i64 20
  %1222 = getelementptr inbounds nuw i8, ptr %1109, i64 24
  %1223 = getelementptr inbounds nuw i8, ptr %1109, i64 28
  %1224 = getelementptr inbounds nuw i8, ptr %1109, i64 32
  %1225 = getelementptr inbounds nuw i8, ptr %1109, i64 36
  %1226 = getelementptr inbounds nuw i8, ptr %1109, i64 40
  %1227 = getelementptr inbounds nuw i8, ptr %1109, i64 44
  %1228 = getelementptr inbounds nuw i8, ptr %1109, i64 48
  %1229 = getelementptr inbounds nuw i8, ptr %1109, i64 52
  %1230 = getelementptr inbounds nuw i8, ptr %1109, i64 56
  %1231 = getelementptr inbounds nuw i8, ptr %1109, i64 60
  %1232 = getelementptr inbounds nuw i8, ptr %1109, i64 64
  %1233 = getelementptr inbounds nuw i8, ptr %1109, i64 68
  %1234 = getelementptr inbounds nuw i8, ptr %1109, i64 72
  %1235 = getelementptr inbounds nuw i8, ptr %1109, i64 76
  %1236 = getelementptr inbounds nuw i8, ptr %1109, i64 80
  %1237 = getelementptr inbounds nuw i8, ptr %1109, i64 84
  %1238 = getelementptr inbounds nuw i8, ptr %1109, i64 88
  %1239 = getelementptr inbounds nuw i8, ptr %1109, i64 92
  %1240 = getelementptr inbounds nuw i8, ptr %1109, i64 96
  %1241 = getelementptr inbounds nuw i8, ptr %1109, i64 100
  %1242 = getelementptr inbounds nuw i8, ptr %1109, i64 104
  %1243 = getelementptr inbounds nuw i8, ptr %1109, i64 108
  %1244 = getelementptr inbounds nuw i8, ptr %1109, i64 112
  %1245 = getelementptr inbounds nuw i8, ptr %1109, i64 116
  %1246 = getelementptr inbounds nuw i8, ptr %1109, i64 120
  %1247 = getelementptr inbounds nuw i8, ptr %1109, i64 124
  %1248 = getelementptr inbounds nuw i8, ptr %1109, i64 128
  %1249 = getelementptr inbounds nuw i8, ptr %1109, i64 132
  %1250 = getelementptr inbounds nuw i8, ptr %1109, i64 136
  %1251 = getelementptr inbounds nuw i8, ptr %1109, i64 140
  %1252 = getelementptr inbounds nuw i8, ptr %1109, i64 144
  %1253 = getelementptr inbounds nuw i8, ptr %1109, i64 148
  %1254 = getelementptr inbounds nuw i8, ptr %1109, i64 152
  %1255 = getelementptr inbounds nuw i8, ptr %1109, i64 156
  %1256 = getelementptr inbounds nuw i8, ptr %1109, i64 160
  %1257 = getelementptr inbounds nuw i8, ptr %1109, i64 164
  %1258 = getelementptr inbounds nuw i8, ptr %1109, i64 168
  %1259 = getelementptr inbounds nuw i8, ptr %1109, i64 172
  %1260 = getelementptr inbounds nuw i8, ptr %1109, i64 176
  %1261 = getelementptr inbounds nuw i8, ptr %1109, i64 180
  %1262 = getelementptr inbounds nuw i8, ptr %1109, i64 184
  %1263 = getelementptr inbounds nuw i8, ptr %1109, i64 188
  %1264 = getelementptr inbounds nuw i8, ptr %1109, i64 192
  %1265 = getelementptr inbounds nuw i8, ptr %1109, i64 196
  %1266 = getelementptr inbounds nuw i8, ptr %1109, i64 200
  %1267 = getelementptr inbounds nuw i8, ptr %1109, i64 204
  %1268 = getelementptr inbounds nuw i8, ptr %1109, i64 208
  %1269 = getelementptr inbounds nuw i8, ptr %1109, i64 212
  %1270 = getelementptr inbounds nuw i8, ptr %1109, i64 216
  %1271 = getelementptr inbounds nuw i8, ptr %1109, i64 220
  %1272 = getelementptr inbounds nuw i8, ptr %1109, i64 224
  %1273 = getelementptr inbounds nuw i8, ptr %1109, i64 228
  %1274 = getelementptr inbounds nuw i8, ptr %1109, i64 232
  %1275 = getelementptr inbounds nuw i8, ptr %1109, i64 236
  %1276 = getelementptr inbounds nuw i8, ptr %1109, i64 240
  %1277 = getelementptr inbounds nuw i8, ptr %1109, i64 244
  %1278 = getelementptr inbounds nuw i8, ptr %1109, i64 248
  %1279 = getelementptr inbounds nuw i8, ptr %1109, i64 252
  %1280 = getelementptr inbounds nuw i8, ptr %1109, i64 256
  %1281 = getelementptr inbounds nuw i8, ptr %1109, i64 260
  %1282 = getelementptr inbounds nuw i8, ptr %1109, i64 264
  %1283 = getelementptr inbounds nuw i8, ptr %1109, i64 268
  %1284 = getelementptr inbounds nuw i8, ptr %1109, i64 272
  %1285 = getelementptr inbounds nuw i8, ptr %1109, i64 276
  %1286 = getelementptr inbounds nuw i8, ptr %1109, i64 280
  %1287 = getelementptr inbounds nuw i8, ptr %1109, i64 284
  %1288 = getelementptr inbounds nuw i8, ptr %1109, i64 288
  %1289 = getelementptr inbounds nuw i8, ptr %1109, i64 292
  %1290 = getelementptr inbounds nuw i8, ptr %1109, i64 296
  %1291 = getelementptr inbounds nuw i8, ptr %1109, i64 300
  %1292 = getelementptr inbounds nuw i8, ptr %1109, i64 304
  %1293 = getelementptr inbounds nuw i8, ptr %1109, i64 308
  %1294 = getelementptr inbounds nuw i8, ptr %1109, i64 312
  %1295 = getelementptr inbounds nuw i8, ptr %1109, i64 316
  %1296 = getelementptr inbounds nuw i8, ptr %1109, i64 320
  %1297 = getelementptr inbounds nuw i8, ptr %1109, i64 324
  %1298 = getelementptr inbounds nuw i8, ptr %1109, i64 328
  %1299 = getelementptr inbounds nuw i8, ptr %1109, i64 332
  %1300 = getelementptr inbounds nuw i8, ptr %1109, i64 336
  %1301 = getelementptr inbounds nuw i8, ptr %1109, i64 340
  %1302 = getelementptr inbounds nuw i8, ptr %1109, i64 344
  %1303 = getelementptr inbounds nuw i8, ptr %1109, i64 348
  %1304 = getelementptr inbounds nuw i8, ptr %1109, i64 352
  %1305 = getelementptr inbounds nuw i8, ptr %1109, i64 356
  %1306 = getelementptr inbounds nuw i8, ptr %1109, i64 360
  %1307 = getelementptr inbounds nuw i8, ptr %1109, i64 364
  %1308 = getelementptr inbounds nuw i8, ptr %1109, i64 368
  %1309 = getelementptr inbounds nuw i8, ptr %1109, i64 372
  %1310 = getelementptr inbounds nuw i8, ptr %1109, i64 376
  %1311 = getelementptr inbounds nuw i8, ptr %1109, i64 380
  br label %1312

1312:                                             ; preds = %1060, %2944
  %1313 = phi double [ 0.000000e+00, %1060 ], [ %2954, %2944 ]
  %1314 = phi float [ %1075, %1060 ], [ %2952, %2944 ]
  %1315 = phi float [ %1074, %1060 ], [ %2948, %2944 ]
  %.pn241419 = phi ptr [ %1110, %1060 ], [ %2955, %2944 ]
  %1316 = phi i64 [ 0, %1060 ], [ %2956, %2944 ]
  store float %1315, ptr %1080, align 64, !tbaa !4
  store float %1314, ptr %1085, align 64, !tbaa !4
  %1317 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 448)
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 64 dereferenceable(384) %1090, i8 0, i64 384, i1 false), !tbaa !4
  %1318 = ptrtoint ptr %1317 to i64
  %1319 = add i64 %1318, 63
  %1320 = and i64 %1319, -64
  %1321 = inttoptr i64 %1320 to ptr
  %1322 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 384)
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(384) %1322, ptr noundef nonnull align 64 dereferenceable(384) %1090, i64 384, i1 false)
  %1323 = shl nuw nsw i64 %1316, 5
  %1324 = add nuw nsw i64 %1323, %.zext
  %1325 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 296
  %1326 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 292
  %1327 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 288
  %1328 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 200
  %1329 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 196
  %1330 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 192
  %1331 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 176
  %1332 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 172
  %1333 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 168
  %1334 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 92
  %1335 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 88
  %1336 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 84
  %1337 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 80
  %1338 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 76
  %1339 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 72
  %1340 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 68
  %1341 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 64
  %1342 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 60
  %1343 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 56
  %1344 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 52
  %1345 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 48
  %1346 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 44
  %1347 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 40
  %1348 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 36
  %1349 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 32
  %1350 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 28
  %1351 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 24
  %1352 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 8
  %1353 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 4
  %1354 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 20
  %1355 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 16
  %1356 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 12
  %1357 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 152
  %1358 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 148
  %1359 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 144
  %1360 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 104
  %1361 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 100
  %1362 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 96
  %1363 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 128
  %1364 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 124
  %1365 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 120
  %1366 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 236
  %1367 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 232
  %1368 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 228
  %1369 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 164
  %1370 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 160
  %1371 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 156
  %1372 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 116
  %1373 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 112
  %1374 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 108
  %1375 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 140
  %1376 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 136
  %1377 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 132
  %1378 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 224
  %1379 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 220
  %1380 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 216
  %1381 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 260
  %1382 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 256
  %1383 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 252
  %1384 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 188
  %1385 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 184
  %1386 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 180
  %1387 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 344
  %1388 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 340
  %1389 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 336
  %1390 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 284
  %1391 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 280
  %1392 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 276
  %1393 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 212
  %1394 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 208
  %1395 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 204
  %1396 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 248
  %1397 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 244
  %1398 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 240
  %1399 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 308
  %1400 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 304
  %1401 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 300
  %1402 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 272
  %1403 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 268
  %1404 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 264
  %1405 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 356
  %1406 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 352
  %1407 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 348
  %1408 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 320
  %1409 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 316
  %1410 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 312
  %1411 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 368
  %1412 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 364
  %1413 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 360
  %1414 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 332
  %1415 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 328
  %1416 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 324
  %1417 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 380
  %1418 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 376
  %1419 = getelementptr inbounds nuw i8, ptr %.pn241419, i64 372
  %1420 = fpext float %1314 to double
  %1421 = fpext float %1315 to double
  %1422 = getelementptr inbounds nuw i8, ptr %1321, i64 4
  %1423 = getelementptr inbounds nuw i8, ptr %1321, i64 8
  %1424 = getelementptr inbounds nuw i8, ptr %1321, i64 12
  %1425 = getelementptr inbounds nuw i8, ptr %1321, i64 16
  %1426 = getelementptr inbounds nuw i8, ptr %1321, i64 20
  %1427 = getelementptr inbounds nuw i8, ptr %1321, i64 24
  %1428 = getelementptr inbounds nuw i8, ptr %1321, i64 28
  %1429 = getelementptr inbounds nuw i8, ptr %1321, i64 32
  %1430 = getelementptr inbounds nuw i8, ptr %1321, i64 36
  %1431 = getelementptr inbounds nuw i8, ptr %1321, i64 40
  %1432 = getelementptr inbounds nuw i8, ptr %1321, i64 44
  %1433 = getelementptr inbounds nuw i8, ptr %1321, i64 48
  %1434 = getelementptr inbounds nuw i8, ptr %1321, i64 52
  %1435 = getelementptr inbounds nuw i8, ptr %1321, i64 56
  %1436 = getelementptr inbounds nuw i8, ptr %1321, i64 60
  %1437 = getelementptr inbounds nuw i8, ptr %1321, i64 64
  %1438 = getelementptr inbounds nuw i8, ptr %1321, i64 68
  %1439 = getelementptr inbounds nuw i8, ptr %1321, i64 72
  %1440 = getelementptr inbounds nuw i8, ptr %1321, i64 76
  %1441 = getelementptr inbounds nuw i8, ptr %1321, i64 80
  %1442 = getelementptr inbounds nuw i8, ptr %1321, i64 84
  %1443 = getelementptr inbounds nuw i8, ptr %1321, i64 88
  %1444 = getelementptr inbounds nuw i8, ptr %1321, i64 92
  %1445 = getelementptr inbounds nuw i8, ptr %1321, i64 96
  %1446 = getelementptr inbounds nuw i8, ptr %1321, i64 100
  %1447 = getelementptr inbounds nuw i8, ptr %1321, i64 104
  %1448 = getelementptr inbounds nuw i8, ptr %1321, i64 108
  %1449 = getelementptr inbounds nuw i8, ptr %1321, i64 112
  %1450 = getelementptr inbounds nuw i8, ptr %1321, i64 116
  %1451 = getelementptr inbounds nuw i8, ptr %1321, i64 120
  %1452 = getelementptr inbounds nuw i8, ptr %1321, i64 124
  %1453 = getelementptr inbounds nuw i8, ptr %1321, i64 128
  %1454 = getelementptr inbounds nuw i8, ptr %1321, i64 132
  %1455 = getelementptr inbounds nuw i8, ptr %1321, i64 136
  %1456 = getelementptr inbounds nuw i8, ptr %1321, i64 140
  %1457 = getelementptr inbounds nuw i8, ptr %1321, i64 144
  %1458 = getelementptr inbounds nuw i8, ptr %1321, i64 148
  %1459 = getelementptr inbounds nuw i8, ptr %1321, i64 152
  %1460 = getelementptr inbounds nuw i8, ptr %1321, i64 156
  %1461 = getelementptr inbounds nuw i8, ptr %1321, i64 160
  %1462 = getelementptr inbounds nuw i8, ptr %1321, i64 164
  %1463 = getelementptr inbounds nuw i8, ptr %1321, i64 168
  %1464 = getelementptr inbounds nuw i8, ptr %1321, i64 172
  %1465 = getelementptr inbounds nuw i8, ptr %1321, i64 176
  %1466 = getelementptr inbounds nuw i8, ptr %1321, i64 180
  %1467 = getelementptr inbounds nuw i8, ptr %1321, i64 184
  %1468 = getelementptr inbounds nuw i8, ptr %1321, i64 188
  %1469 = getelementptr inbounds nuw i8, ptr %1321, i64 192
  %1470 = getelementptr inbounds nuw i8, ptr %1321, i64 196
  %1471 = getelementptr inbounds nuw i8, ptr %1321, i64 200
  %1472 = getelementptr inbounds nuw i8, ptr %1321, i64 204
  %1473 = getelementptr inbounds nuw i8, ptr %1321, i64 208
  %1474 = getelementptr inbounds nuw i8, ptr %1321, i64 212
  %1475 = getelementptr inbounds nuw i8, ptr %1321, i64 216
  %1476 = getelementptr inbounds nuw i8, ptr %1321, i64 220
  %1477 = getelementptr inbounds nuw i8, ptr %1321, i64 224
  %1478 = getelementptr inbounds nuw i8, ptr %1321, i64 228
  %1479 = getelementptr inbounds nuw i8, ptr %1321, i64 232
  %1480 = getelementptr inbounds nuw i8, ptr %1321, i64 236
  %1481 = getelementptr inbounds nuw i8, ptr %1321, i64 240
  %1482 = getelementptr inbounds nuw i8, ptr %1321, i64 244
  %1483 = getelementptr inbounds nuw i8, ptr %1321, i64 248
  %1484 = getelementptr inbounds nuw i8, ptr %1321, i64 252
  %1485 = getelementptr inbounds nuw i8, ptr %1321, i64 256
  %1486 = getelementptr inbounds nuw i8, ptr %1321, i64 260
  %1487 = getelementptr inbounds nuw i8, ptr %1321, i64 264
  %1488 = getelementptr inbounds nuw i8, ptr %1321, i64 268
  %1489 = getelementptr inbounds nuw i8, ptr %1321, i64 272
  %1490 = getelementptr inbounds nuw i8, ptr %1321, i64 276
  %1491 = getelementptr inbounds nuw i8, ptr %1321, i64 280
  %1492 = getelementptr inbounds nuw i8, ptr %1321, i64 284
  %1493 = getelementptr inbounds nuw i8, ptr %1321, i64 288
  %1494 = getelementptr inbounds nuw i8, ptr %1321, i64 292
  %1495 = getelementptr inbounds nuw i8, ptr %1321, i64 296
  %1496 = getelementptr inbounds nuw i8, ptr %1321, i64 300
  %1497 = getelementptr inbounds nuw i8, ptr %1321, i64 304
  %1498 = getelementptr inbounds nuw i8, ptr %1321, i64 308
  %1499 = getelementptr inbounds nuw i8, ptr %1321, i64 312
  %1500 = getelementptr inbounds nuw i8, ptr %1321, i64 316
  %1501 = getelementptr inbounds nuw i8, ptr %1321, i64 320
  %1502 = getelementptr inbounds nuw i8, ptr %1321, i64 324
  %1503 = getelementptr inbounds nuw i8, ptr %1321, i64 328
  %1504 = getelementptr inbounds nuw i8, ptr %1321, i64 332
  %1505 = getelementptr inbounds nuw i8, ptr %1321, i64 336
  %1506 = getelementptr inbounds nuw i8, ptr %1321, i64 340
  %1507 = getelementptr inbounds nuw i8, ptr %1321, i64 344
  %1508 = getelementptr inbounds nuw i8, ptr %1321, i64 348
  %1509 = getelementptr inbounds nuw i8, ptr %1321, i64 352
  %1510 = getelementptr inbounds nuw i8, ptr %1321, i64 356
  %1511 = getelementptr inbounds nuw i8, ptr %1321, i64 360
  %1512 = getelementptr inbounds nuw i8, ptr %1321, i64 364
  %1513 = getelementptr inbounds nuw i8, ptr %1321, i64 368
  %1514 = getelementptr inbounds nuw i8, ptr %1321, i64 372
  %1515 = getelementptr inbounds nuw i8, ptr %1321, i64 376
  %1516 = getelementptr inbounds nuw i8, ptr %1321, i64 380
  br label %.preheader332

.preheader332:                                    ; preds = %1312, %.preheader332
  %1517 = phi double [ 0.000000e+00, %1312 ], [ %2209, %.preheader332 ]
  %1518 = phi double [ 0.000000e+00, %1312 ], [ %2208, %.preheader332 ]
  %1519 = phi double [ 0.000000e+00, %1312 ], [ %2206, %.preheader332 ]
  %.pn297418 = phi ptr [ %1322, %1312 ], [ %2215, %.preheader332 ]
  %1520 = phi i64 [ 0, %1312 ], [ %2216, %.preheader332 ]
  %1521 = add nuw nsw i64 %1324, %1520
  %1522 = urem i64 %1521, 96
  %1523 = shl nuw nsw i64 %1522, 3
  %1524 = getelementptr inbounds nuw float, ptr %31, i64 %1522
  %1525 = load float, ptr %1524, align 4, !tbaa !4
  %1526 = getelementptr inbounds nuw float, ptr %36, i64 %1522
  %1527 = load float, ptr %1526, align 4, !tbaa !4
  %1528 = getelementptr inbounds nuw float, ptr %24, i64 %1523
  %1529 = load float, ptr %1528, align 4, !tbaa !4
  store float %1529, ptr %1095, align 64, !tbaa !4
  %1530 = getelementptr inbounds nuw i8, ptr %1528, i64 4
  %1531 = load float, ptr %1530, align 4, !tbaa !4
  store float %1531, ptr %1115, align 4, !tbaa !4
  %1532 = getelementptr inbounds nuw i8, ptr %1528, i64 8
  %1533 = load float, ptr %1532, align 4, !tbaa !4
  store float %1533, ptr %1116, align 8, !tbaa !4
  %1534 = getelementptr inbounds nuw i8, ptr %1528, i64 12
  %1535 = load float, ptr %1534, align 4, !tbaa !4
  store float %1535, ptr %1117, align 4, !tbaa !4
  %1536 = getelementptr inbounds nuw i8, ptr %1528, i64 16
  %1537 = load float, ptr %1536, align 4, !tbaa !4
  store float %1537, ptr %1118, align 16, !tbaa !4
  %1538 = getelementptr inbounds nuw i8, ptr %1528, i64 20
  %1539 = load float, ptr %1538, align 4, !tbaa !4
  store float %1539, ptr %1119, align 4, !tbaa !4
  %1540 = getelementptr inbounds nuw i8, ptr %1528, i64 24
  %1541 = load float, ptr %1540, align 4, !tbaa !4
  store float %1541, ptr %1120, align 8, !tbaa !4
  %1542 = getelementptr inbounds nuw i8, ptr %1528, i64 28
  %1543 = load float, ptr %1542, align 4, !tbaa !4
  store float %1543, ptr %1121, align 4, !tbaa !4
  %1544 = load float, ptr %1325, align 4, !tbaa !4
  %1545 = load float, ptr %1326, align 4, !tbaa !4
  %1546 = load float, ptr %1327, align 4, !tbaa !4
  %1547 = load float, ptr %1328, align 4, !tbaa !4
  %1548 = load float, ptr %1329, align 4, !tbaa !4
  %1549 = load float, ptr %1330, align 4, !tbaa !4
  %1550 = load float, ptr %1331, align 4, !tbaa !4
  %1551 = load float, ptr %1332, align 4, !tbaa !4
  %1552 = load float, ptr %1333, align 4, !tbaa !4
  %1553 = load float, ptr %1334, align 4, !tbaa !4
  %1554 = load float, ptr %1335, align 4, !tbaa !4
  %1555 = load float, ptr %1336, align 4, !tbaa !4
  %1556 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 96)
  %1557 = ptrtoint ptr %1556 to i64
  %1558 = add i64 %1557, 63
  %1559 = and i64 %1558, -64
  %1560 = inttoptr i64 %1559 to ptr
  store float 0x400921FB60000000, ptr %1560, align 64, !tbaa !4
  %1561 = getelementptr inbounds nuw i8, ptr %1560, i64 4
  store float 0x400921FB60000000, ptr %1561, align 4, !tbaa !4
  %1562 = getelementptr inbounds nuw i8, ptr %1560, i64 8
  store float 0x400921FB60000000, ptr %1562, align 8, !tbaa !4
  %1563 = getelementptr inbounds nuw i8, ptr %1560, i64 12
  store float 0x400921FB60000000, ptr %1563, align 4, !tbaa !4
  %1564 = getelementptr inbounds nuw i8, ptr %1560, i64 16
  store float 0x400921FB60000000, ptr %1564, align 16, !tbaa !4
  %1565 = getelementptr inbounds nuw i8, ptr %1560, i64 20
  store float 0x400921FB60000000, ptr %1565, align 4, !tbaa !4
  %1566 = getelementptr inbounds nuw i8, ptr %1560, i64 24
  store float 0x400921FB60000000, ptr %1566, align 8, !tbaa !4
  %1567 = getelementptr inbounds nuw i8, ptr %1560, i64 28
  store float 0x400921FB60000000, ptr %1567, align 4, !tbaa !4
  %1568 = load float, ptr %1095, align 64, !tbaa !4
  %1569 = fmul float %1568, 0x400921FB60000000
  store float %1569, ptr %1560, align 64, !tbaa !4
  %1570 = load float, ptr %1115, align 4, !tbaa !4
  %1571 = fmul float %1570, 0x400921FB60000000
  store float %1571, ptr %1561, align 4, !tbaa !4
  %1572 = load float, ptr %1116, align 8, !tbaa !4
  %1573 = fmul float %1572, 0x400921FB60000000
  store float %1573, ptr %1562, align 8, !tbaa !4
  %1574 = load float, ptr %1117, align 4, !tbaa !4
  %1575 = fmul float %1574, 0x400921FB60000000
  store float %1575, ptr %1563, align 4, !tbaa !4
  %1576 = load float, ptr %1118, align 16, !tbaa !4
  %1577 = fmul float %1576, 0x400921FB60000000
  store float %1577, ptr %1564, align 16, !tbaa !4
  %1578 = load float, ptr %1119, align 4, !tbaa !4
  %1579 = fmul float %1578, 0x400921FB60000000
  store float %1579, ptr %1565, align 4, !tbaa !4
  %1580 = load float, ptr %1120, align 8, !tbaa !4
  %1581 = fmul float %1580, 0x400921FB60000000
  store float %1581, ptr %1566, align 8, !tbaa !4
  %1582 = load float, ptr %1121, align 4, !tbaa !4
  %1583 = fmul float %1582, 0x400921FB60000000
  store float %1583, ptr %1567, align 4, !tbaa !4
  tail call void @__catalyst__rt__device_init(ptr nonnull @"/home/pichau/QAgents/.venv/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.so", ptr nonnull @LightningSimulator, ptr nonnull @"{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}", i64 0, i1 false)
  %1584 = tail call ptr @__catalyst__rt__qubit_allocate_array(i64 8)
  %1585 = tail call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %1584, i64 7)
  %1586 = load ptr, ptr %1585, align 8
  %1587 = fpext float %1583 to double
  tail call void @__catalyst__qis__RY(double %1587, ptr %1586, ptr null)
  %1588 = fpext float %1555 to double
  tail call void @__catalyst__qis__RZ(double %1588, ptr %1586, ptr null)
  %1589 = fpext float %1554 to double
  tail call void @__catalyst__qis__RY(double %1589, ptr %1586, ptr null)
  %1590 = fpext float %1553 to double
  tail call void @__catalyst__qis__RZ(double %1590, ptr %1586, ptr null)
  %1591 = load float, ptr %1337, align 4, !tbaa !4
  %1592 = load float, ptr %1338, align 4, !tbaa !4
  %1593 = load float, ptr %1339, align 4, !tbaa !4
  %1594 = load float, ptr %1566, align 8, !tbaa !4
  %1595 = tail call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %1584, i64 6)
  %1596 = load ptr, ptr %1595, align 8
  %1597 = fpext float %1594 to double
  tail call void @__catalyst__qis__RY(double %1597, ptr %1596, ptr null)
  %1598 = fpext float %1593 to double
  tail call void @__catalyst__qis__RZ(double %1598, ptr %1596, ptr null)
  %1599 = fpext float %1592 to double
  tail call void @__catalyst__qis__RY(double %1599, ptr %1596, ptr null)
  %1600 = fpext float %1591 to double
  tail call void @__catalyst__qis__RZ(double %1600, ptr %1596, ptr null)
  %1601 = load float, ptr %1340, align 4, !tbaa !4
  %1602 = load float, ptr %1341, align 4, !tbaa !4
  %1603 = load float, ptr %1342, align 4, !tbaa !4
  %1604 = load float, ptr %1565, align 4, !tbaa !4
  %1605 = tail call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %1584, i64 5)
  %1606 = load ptr, ptr %1605, align 8
  %1607 = fpext float %1604 to double
  tail call void @__catalyst__qis__RY(double %1607, ptr %1606, ptr null)
  %1608 = fpext float %1603 to double
  tail call void @__catalyst__qis__RZ(double %1608, ptr %1606, ptr null)
  %1609 = fpext float %1602 to double
  tail call void @__catalyst__qis__RY(double %1609, ptr %1606, ptr null)
  %1610 = fpext float %1601 to double
  tail call void @__catalyst__qis__RZ(double %1610, ptr %1606, ptr null)
  %1611 = load float, ptr %1343, align 4, !tbaa !4
  %1612 = load float, ptr %1344, align 4, !tbaa !4
  %1613 = load float, ptr %1345, align 4, !tbaa !4
  %1614 = load float, ptr %1564, align 16, !tbaa !4
  %1615 = tail call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %1584, i64 4)
  %1616 = load ptr, ptr %1615, align 8
  %1617 = fpext float %1614 to double
  tail call void @__catalyst__qis__RY(double %1617, ptr %1616, ptr null)
  %1618 = fpext float %1613 to double
  tail call void @__catalyst__qis__RZ(double %1618, ptr %1616, ptr null)
  %1619 = fpext float %1612 to double
  tail call void @__catalyst__qis__RY(double %1619, ptr %1616, ptr null)
  %1620 = fpext float %1611 to double
  tail call void @__catalyst__qis__RZ(double %1620, ptr %1616, ptr null)
  %1621 = load float, ptr %1346, align 4, !tbaa !4
  %1622 = load float, ptr %1347, align 4, !tbaa !4
  %1623 = load float, ptr %1348, align 4, !tbaa !4
  %1624 = load float, ptr %1563, align 4, !tbaa !4
  %1625 = tail call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %1584, i64 3)
  %1626 = load ptr, ptr %1625, align 8
  %1627 = fpext float %1624 to double
  tail call void @__catalyst__qis__RY(double %1627, ptr %1626, ptr null)
  %1628 = fpext float %1623 to double
  tail call void @__catalyst__qis__RZ(double %1628, ptr %1626, ptr null)
  %1629 = fpext float %1622 to double
  tail call void @__catalyst__qis__RY(double %1629, ptr %1626, ptr null)
  %1630 = fpext float %1621 to double
  tail call void @__catalyst__qis__RZ(double %1630, ptr %1626, ptr null)
  %1631 = load float, ptr %1349, align 4, !tbaa !4
  %1632 = load float, ptr %1350, align 4, !tbaa !4
  %1633 = load float, ptr %1351, align 4, !tbaa !4
  %1634 = load float, ptr %1562, align 8, !tbaa !4
  %1635 = tail call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %1584, i64 2)
  %1636 = load ptr, ptr %1635, align 8
  %1637 = fpext float %1634 to double
  tail call void @__catalyst__qis__RY(double %1637, ptr %1636, ptr null)
  %1638 = fpext float %1633 to double
  tail call void @__catalyst__qis__RZ(double %1638, ptr %1636, ptr null)
  %1639 = fpext float %1632 to double
  tail call void @__catalyst__qis__RY(double %1639, ptr %1636, ptr null)
  %1640 = fpext float %1631 to double
  tail call void @__catalyst__qis__RZ(double %1640, ptr %1636, ptr null)
  %1641 = load float, ptr %1352, align 4, !tbaa !4
  %1642 = load float, ptr %1353, align 4, !tbaa !4
  %1643 = load float, ptr %.pn241419, align 4, !tbaa !4
  %1644 = load float, ptr %1560, align 64, !tbaa !4
  %1645 = tail call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %1584, i64 0)
  %1646 = load ptr, ptr %1645, align 8
  %1647 = fpext float %1644 to double
  tail call void @__catalyst__qis__RY(double %1647, ptr %1646, ptr null)
  %1648 = fpext float %1643 to double
  tail call void @__catalyst__qis__RZ(double %1648, ptr %1646, ptr null)
  %1649 = fpext float %1642 to double
  tail call void @__catalyst__qis__RY(double %1649, ptr %1646, ptr null)
  %1650 = fpext float %1641 to double
  tail call void @__catalyst__qis__RZ(double %1650, ptr %1646, ptr null)
  %1651 = load float, ptr %1354, align 4, !tbaa !4
  %1652 = load float, ptr %1355, align 4, !tbaa !4
  %1653 = load float, ptr %1356, align 4, !tbaa !4
  %1654 = load float, ptr %1561, align 4, !tbaa !4
  tail call void @_mlir_memref_to_llvm_free(ptr %1556)
  %1655 = tail call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %1584, i64 1)
  %1656 = load ptr, ptr %1655, align 8
  %1657 = fpext float %1654 to double
  tail call void @__catalyst__qis__RY(double %1657, ptr %1656, ptr null)
  %1658 = fpext float %1653 to double
  tail call void @__catalyst__qis__RZ(double %1658, ptr %1656, ptr null)
  %1659 = fpext float %1652 to double
  tail call void @__catalyst__qis__RY(double %1659, ptr %1656, ptr null)
  %1660 = fpext float %1651 to double
  tail call void @__catalyst__qis__RZ(double %1660, ptr %1656, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %1646, ptr %1656, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %1656, ptr %1636, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %1636, ptr %1626, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %1626, ptr %1616, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %1616, ptr %1606, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %1606, ptr %1596, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %1596, ptr %1586, ptr null)
  %1661 = fpext float %1552 to double
  tail call void @__catalyst__qis__RZ(double %1661, ptr %1596, ptr null)
  %1662 = fpext float %1551 to double
  tail call void @__catalyst__qis__RY(double %1662, ptr %1596, ptr null)
  %1663 = fpext float %1550 to double
  tail call void @__catalyst__qis__RZ(double %1663, ptr %1596, ptr null)
  %1664 = load float, ptr %1357, align 4, !tbaa !4
  %1665 = load float, ptr %1358, align 4, !tbaa !4
  %1666 = load float, ptr %1359, align 4, !tbaa !4
  %1667 = fpext float %1666 to double
  tail call void @__catalyst__qis__RZ(double %1667, ptr %1616, ptr null)
  %1668 = fpext float %1665 to double
  tail call void @__catalyst__qis__RY(double %1668, ptr %1616, ptr null)
  %1669 = fpext float %1664 to double
  tail call void @__catalyst__qis__RZ(double %1669, ptr %1616, ptr null)
  %1670 = load float, ptr %1360, align 4, !tbaa !4
  %1671 = load float, ptr %1361, align 4, !tbaa !4
  %1672 = load float, ptr %1362, align 4, !tbaa !4
  tail call void @__catalyst__qis__CNOT(ptr %1586, ptr %1646, ptr null)
  %1673 = fpext float %1672 to double
  tail call void @__catalyst__qis__RZ(double %1673, ptr %1646, ptr null)
  %1674 = fpext float %1671 to double
  tail call void @__catalyst__qis__RY(double %1674, ptr %1646, ptr null)
  %1675 = fpext float %1670 to double
  tail call void @__catalyst__qis__RZ(double %1675, ptr %1646, ptr null)
  %1676 = load float, ptr %1363, align 4, !tbaa !4
  %1677 = load float, ptr %1364, align 4, !tbaa !4
  %1678 = load float, ptr %1365, align 4, !tbaa !4
  %1679 = fpext float %1678 to double
  tail call void @__catalyst__qis__RZ(double %1679, ptr %1636, ptr null)
  %1680 = fpext float %1677 to double
  tail call void @__catalyst__qis__RY(double %1680, ptr %1636, ptr null)
  %1681 = fpext float %1676 to double
  tail call void @__catalyst__qis__RZ(double %1681, ptr %1636, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %1646, ptr %1636, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %1636, ptr %1616, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %1616, ptr %1596, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %1596, ptr %1646, ptr null)
  %1682 = fpext float %1549 to double
  tail call void @__catalyst__qis__RZ(double %1682, ptr %1646, ptr null)
  %1683 = fpext float %1548 to double
  tail call void @__catalyst__qis__RY(double %1683, ptr %1646, ptr null)
  %1684 = fpext float %1547 to double
  tail call void @__catalyst__qis__RZ(double %1684, ptr %1646, ptr null)
  %1685 = load float, ptr %1366, align 4, !tbaa !4
  %1686 = load float, ptr %1367, align 4, !tbaa !4
  %1687 = load float, ptr %1368, align 4, !tbaa !4
  %1688 = load float, ptr %1369, align 4, !tbaa !4
  %1689 = load float, ptr %1370, align 4, !tbaa !4
  %1690 = load float, ptr %1371, align 4, !tbaa !4
  %1691 = fpext float %1690 to double
  tail call void @__catalyst__qis__RZ(double %1691, ptr %1606, ptr null)
  %1692 = fpext float %1689 to double
  tail call void @__catalyst__qis__RY(double %1692, ptr %1606, ptr null)
  %1693 = fpext float %1688 to double
  tail call void @__catalyst__qis__RZ(double %1693, ptr %1606, ptr null)
  %1694 = load float, ptr %1372, align 4, !tbaa !4
  %1695 = load float, ptr %1373, align 4, !tbaa !4
  %1696 = load float, ptr %1374, align 4, !tbaa !4
  %1697 = fpext float %1696 to double
  tail call void @__catalyst__qis__RZ(double %1697, ptr %1656, ptr null)
  %1698 = fpext float %1695 to double
  tail call void @__catalyst__qis__RY(double %1698, ptr %1656, ptr null)
  %1699 = fpext float %1694 to double
  tail call void @__catalyst__qis__RZ(double %1699, ptr %1656, ptr null)
  %1700 = load float, ptr %1375, align 4, !tbaa !4
  %1701 = load float, ptr %1376, align 4, !tbaa !4
  %1702 = load float, ptr %1377, align 4, !tbaa !4
  %1703 = fpext float %1702 to double
  tail call void @__catalyst__qis__RZ(double %1703, ptr %1626, ptr null)
  %1704 = fpext float %1701 to double
  tail call void @__catalyst__qis__RY(double %1704, ptr %1626, ptr null)
  %1705 = fpext float %1700 to double
  tail call void @__catalyst__qis__RZ(double %1705, ptr %1626, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %1656, ptr %1626, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %1626, ptr %1606, ptr null)
  %1706 = fpext float %1687 to double
  tail call void @__catalyst__qis__RZ(double %1706, ptr %1626, ptr null)
  %1707 = fpext float %1686 to double
  tail call void @__catalyst__qis__RY(double %1707, ptr %1626, ptr null)
  %1708 = fpext float %1685 to double
  tail call void @__catalyst__qis__RZ(double %1708, ptr %1626, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %1646, ptr %1626, ptr null)
  %1709 = load float, ptr %1378, align 4, !tbaa !4
  %1710 = load float, ptr %1379, align 4, !tbaa !4
  %1711 = load float, ptr %1380, align 4, !tbaa !4
  %1712 = fpext float %1711 to double
  tail call void @__catalyst__qis__RZ(double %1712, ptr %1636, ptr null)
  %1713 = fpext float %1710 to double
  tail call void @__catalyst__qis__RY(double %1713, ptr %1636, ptr null)
  %1714 = fpext float %1709 to double
  tail call void @__catalyst__qis__RZ(double %1714, ptr %1636, ptr null)
  %1715 = load float, ptr %1381, align 4, !tbaa !4
  %1716 = load float, ptr %1382, align 4, !tbaa !4
  %1717 = load float, ptr %1383, align 4, !tbaa !4
  %1718 = load float, ptr %1384, align 4, !tbaa !4
  %1719 = load float, ptr %1385, align 4, !tbaa !4
  %1720 = load float, ptr %1386, align 4, !tbaa !4
  %1721 = fpext float %1720 to double
  tail call void @__catalyst__qis__RZ(double %1721, ptr %1586, ptr null)
  %1722 = fpext float %1719 to double
  tail call void @__catalyst__qis__RY(double %1722, ptr %1586, ptr null)
  %1723 = fpext float %1718 to double
  tail call void @__catalyst__qis__RZ(double %1723, ptr %1586, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %1606, ptr %1586, ptr null)
  %1724 = fpext float %1717 to double
  tail call void @__catalyst__qis__RZ(double %1724, ptr %1606, ptr null)
  %1725 = fpext float %1716 to double
  tail call void @__catalyst__qis__RY(double %1725, ptr %1606, ptr null)
  %1726 = fpext float %1715 to double
  tail call void @__catalyst__qis__RZ(double %1726, ptr %1606, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %1636, ptr %1606, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %1606, ptr %1646, ptr null)
  %1727 = fpext float %1546 to double
  tail call void @__catalyst__qis__RZ(double %1727, ptr %1646, ptr null)
  %1728 = fpext float %1545 to double
  tail call void @__catalyst__qis__RY(double %1728, ptr %1646, ptr null)
  %1729 = fpext float %1544 to double
  tail call void @__catalyst__qis__RZ(double %1729, ptr %1646, ptr null)
  %1730 = load float, ptr %1387, align 4, !tbaa !4
  %1731 = load float, ptr %1388, align 4, !tbaa !4
  %1732 = load float, ptr %1389, align 4, !tbaa !4
  %1733 = load float, ptr %1390, align 4, !tbaa !4
  %1734 = load float, ptr %1391, align 4, !tbaa !4
  %1735 = load float, ptr %1392, align 4, !tbaa !4
  tail call void @__catalyst__qis__CNOT(ptr %1586, ptr %1656, ptr null)
  %1736 = fpext float %1735 to double
  tail call void @__catalyst__qis__RZ(double %1736, ptr %1586, ptr null)
  %1737 = fpext float %1734 to double
  tail call void @__catalyst__qis__RY(double %1737, ptr %1586, ptr null)
  %1738 = fpext float %1733 to double
  tail call void @__catalyst__qis__RZ(double %1738, ptr %1586, ptr null)
  %1739 = load float, ptr %1393, align 4, !tbaa !4
  %1740 = load float, ptr %1394, align 4, !tbaa !4
  %1741 = load float, ptr %1395, align 4, !tbaa !4
  %1742 = fpext float %1741 to double
  tail call void @__catalyst__qis__RZ(double %1742, ptr %1656, ptr null)
  %1743 = fpext float %1740 to double
  tail call void @__catalyst__qis__RY(double %1743, ptr %1656, ptr null)
  %1744 = fpext float %1739 to double
  tail call void @__catalyst__qis__RZ(double %1744, ptr %1656, ptr null)
  %1745 = load float, ptr %1396, align 4, !tbaa !4
  %1746 = load float, ptr %1397, align 4, !tbaa !4
  %1747 = load float, ptr %1398, align 4, !tbaa !4
  %1748 = fpext float %1747 to double
  tail call void @__catalyst__qis__RZ(double %1748, ptr %1616, ptr null)
  %1749 = fpext float %1746 to double
  tail call void @__catalyst__qis__RY(double %1749, ptr %1616, ptr null)
  %1750 = fpext float %1745 to double
  tail call void @__catalyst__qis__RZ(double %1750, ptr %1616, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %1656, ptr %1616, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %1616, ptr %1586, ptr null)
  %1751 = fpext float %1732 to double
  tail call void @__catalyst__qis__RZ(double %1751, ptr %1616, ptr null)
  %1752 = fpext float %1731 to double
  tail call void @__catalyst__qis__RY(double %1752, ptr %1616, ptr null)
  %1753 = fpext float %1730 to double
  tail call void @__catalyst__qis__RZ(double %1753, ptr %1616, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %1646, ptr %1616, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %1616, ptr %1646, ptr null)
  %1754 = load float, ptr %1399, align 4, !tbaa !4
  %1755 = load float, ptr %1400, align 4, !tbaa !4
  %1756 = load float, ptr %1401, align 4, !tbaa !4
  %1757 = load float, ptr %1402, align 4, !tbaa !4
  %1758 = load float, ptr %1403, align 4, !tbaa !4
  %1759 = load float, ptr %1404, align 4, !tbaa !4
  %1760 = fpext float %1759 to double
  tail call void @__catalyst__qis__RZ(double %1760, ptr %1596, ptr null)
  %1761 = fpext float %1758 to double
  tail call void @__catalyst__qis__RY(double %1761, ptr %1596, ptr null)
  %1762 = fpext float %1757 to double
  tail call void @__catalyst__qis__RZ(double %1762, ptr %1596, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %1626, ptr %1596, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %1596, ptr %1656, ptr null)
  %1763 = fpext float %1756 to double
  tail call void @__catalyst__qis__RZ(double %1763, ptr %1656, ptr null)
  %1764 = fpext float %1755 to double
  tail call void @__catalyst__qis__RY(double %1764, ptr %1656, ptr null)
  %1765 = fpext float %1754 to double
  tail call void @__catalyst__qis__RZ(double %1765, ptr %1656, ptr null)
  %1766 = load float, ptr %1405, align 4, !tbaa !4
  %1767 = load float, ptr %1406, align 4, !tbaa !4
  %1768 = load float, ptr %1407, align 4, !tbaa !4
  %1769 = fpext float %1768 to double
  tail call void @__catalyst__qis__RZ(double %1769, ptr %1606, ptr null)
  %1770 = fpext float %1767 to double
  tail call void @__catalyst__qis__RY(double %1770, ptr %1606, ptr null)
  %1771 = fpext float %1766 to double
  tail call void @__catalyst__qis__RZ(double %1771, ptr %1606, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %1656, ptr %1606, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %1606, ptr %1656, ptr null)
  %1772 = load float, ptr %1408, align 4, !tbaa !4
  %1773 = load float, ptr %1409, align 4, !tbaa !4
  %1774 = load float, ptr %1410, align 4, !tbaa !4
  tail call void @__catalyst__qis__CNOT(ptr %1586, ptr %1636, ptr null)
  %1775 = fpext float %1774 to double
  tail call void @__catalyst__qis__RZ(double %1775, ptr %1636, ptr null)
  %1776 = fpext float %1773 to double
  tail call void @__catalyst__qis__RY(double %1776, ptr %1636, ptr null)
  %1777 = fpext float %1772 to double
  tail call void @__catalyst__qis__RZ(double %1777, ptr %1636, ptr null)
  %1778 = load float, ptr %1411, align 4, !tbaa !4
  %1779 = load float, ptr %1412, align 4, !tbaa !4
  %1780 = load float, ptr %1413, align 4, !tbaa !4
  %1781 = fpext float %1780 to double
  tail call void @__catalyst__qis__RZ(double %1781, ptr %1596, ptr null)
  %1782 = fpext float %1779 to double
  tail call void @__catalyst__qis__RY(double %1782, ptr %1596, ptr null)
  %1783 = fpext float %1778 to double
  tail call void @__catalyst__qis__RZ(double %1783, ptr %1596, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %1636, ptr %1596, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %1596, ptr %1636, ptr null)
  %1784 = load float, ptr %1414, align 4, !tbaa !4
  %1785 = load float, ptr %1415, align 4, !tbaa !4
  %1786 = load float, ptr %1416, align 4, !tbaa !4
  %1787 = fpext float %1786 to double
  tail call void @__catalyst__qis__RZ(double %1787, ptr %1626, ptr null)
  %1788 = fpext float %1785 to double
  tail call void @__catalyst__qis__RY(double %1788, ptr %1626, ptr null)
  %1789 = fpext float %1784 to double
  tail call void @__catalyst__qis__RZ(double %1789, ptr %1626, ptr null)
  %1790 = load float, ptr %1417, align 4, !tbaa !4
  %1791 = load float, ptr %1418, align 4, !tbaa !4
  %1792 = load float, ptr %1419, align 4, !tbaa !4
  %1793 = fpext float %1792 to double
  tail call void @__catalyst__qis__RZ(double %1793, ptr %1586, ptr null)
  %1794 = fpext float %1791 to double
  tail call void @__catalyst__qis__RY(double %1794, ptr %1586, ptr null)
  %1795 = fpext float %1790 to double
  tail call void @__catalyst__qis__RZ(double %1795, ptr %1586, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %1626, ptr %1586, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %1586, ptr %1626, ptr null)
  %1796 = tail call i64 @__catalyst__qis__NamedObs(i64 3, ptr %1646)
  %1797 = tail call double @__catalyst__qis__Expval(i64 %1796)
  %1798 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %1799 = ptrtoint ptr %1798 to i64
  %1800 = add i64 %1799, 63
  %1801 = and i64 %1800, -64
  %1802 = inttoptr i64 %1801 to ptr
  store double %1797, ptr %1802, align 64, !tbaa !6
  tail call void @__catalyst__rt__qubit_release_array(ptr %1584)
  tail call void @__catalyst__rt__device_release()
  %1803 = load double, ptr %1802, align 64, !tbaa !6
  store double 1.000000e+00, ptr %1100, align 64, !tbaa !6
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 1 dereferenceable(384) %1101, i8 0, i64 384, i1 false)
  store i32 0, ptr %1102, align 1
  store i32 0, ptr %1103, align 1
  tail call void (...) @__enzyme_autodiff0(ptr nonnull @_sample_loss.cloned, ptr nonnull @enzyme_const, ptr %.pn241419, ptr %.pn241419, ptr nonnull %1101, i64 0, i64 4, i64 8, i64 3, i64 24, i64 3, i64 1, ptr nonnull @enzyme_const, ptr %1076, ptr nonnull %1080, ptr nonnull %1102, i64 0, ptr nonnull @enzyme_const, ptr %1081, ptr nonnull %1085, ptr nonnull %1103, i64 0, ptr nonnull @enzyme_const, ptr %23, ptr nonnull @enzyme_const, ptr %24, i64 %1523, i64 8, i64 1, ptr nonnull @enzyme_const, ptr %30, ptr nonnull @enzyme_const, ptr %31, i64 %1522, ptr nonnull @enzyme_const, ptr %35, ptr nonnull @enzyme_const, ptr %36, i64 %1522, ptr nonnull @enzyme_const, ptr %1104, ptr nonnull @enzyme_dupnoneed, ptr %1104, ptr nonnull %1100, i64 0)
  %1804 = load float, ptr %1103, align 4, !tbaa !4
  %1805 = load float, ptr %1102, align 4, !tbaa !4
  %1806 = load float, ptr %.pn297418, align 4, !tbaa !4
  %1807 = load float, ptr %1101, align 4, !tbaa !4
  %1808 = fadd float %1806, %1807
  store float %1808, ptr %1321, align 64, !tbaa !4
  %1809 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 4
  %1810 = load float, ptr %1809, align 4, !tbaa !4
  %1811 = load float, ptr %1122, align 4, !tbaa !4
  %1812 = fadd float %1810, %1811
  store float %1812, ptr %1422, align 4, !tbaa !4
  %1813 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 8
  %1814 = load float, ptr %1813, align 4, !tbaa !4
  %1815 = load float, ptr %1123, align 4, !tbaa !4
  %1816 = fadd float %1814, %1815
  store float %1816, ptr %1423, align 8, !tbaa !4
  %1817 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 12
  %1818 = load float, ptr %1817, align 4, !tbaa !4
  %1819 = load float, ptr %1124, align 4, !tbaa !4
  %1820 = fadd float %1818, %1819
  store float %1820, ptr %1424, align 4, !tbaa !4
  %1821 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 16
  %1822 = load float, ptr %1821, align 4, !tbaa !4
  %1823 = load float, ptr %1125, align 4, !tbaa !4
  %1824 = fadd float %1822, %1823
  store float %1824, ptr %1425, align 16, !tbaa !4
  %1825 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 20
  %1826 = load float, ptr %1825, align 4, !tbaa !4
  %1827 = load float, ptr %1126, align 4, !tbaa !4
  %1828 = fadd float %1826, %1827
  store float %1828, ptr %1426, align 4, !tbaa !4
  %1829 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 24
  %1830 = load float, ptr %1829, align 4, !tbaa !4
  %1831 = load float, ptr %1127, align 4, !tbaa !4
  %1832 = fadd float %1830, %1831
  store float %1832, ptr %1427, align 8, !tbaa !4
  %1833 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 28
  %1834 = load float, ptr %1833, align 4, !tbaa !4
  %1835 = load float, ptr %1128, align 4, !tbaa !4
  %1836 = fadd float %1834, %1835
  store float %1836, ptr %1428, align 4, !tbaa !4
  %1837 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 32
  %1838 = load float, ptr %1837, align 4, !tbaa !4
  %1839 = load float, ptr %1129, align 4, !tbaa !4
  %1840 = fadd float %1838, %1839
  store float %1840, ptr %1429, align 32, !tbaa !4
  %1841 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 36
  %1842 = load float, ptr %1841, align 4, !tbaa !4
  %1843 = load float, ptr %1130, align 4, !tbaa !4
  %1844 = fadd float %1842, %1843
  store float %1844, ptr %1430, align 4, !tbaa !4
  %1845 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 40
  %1846 = load float, ptr %1845, align 4, !tbaa !4
  %1847 = load float, ptr %1131, align 4, !tbaa !4
  %1848 = fadd float %1846, %1847
  store float %1848, ptr %1431, align 8, !tbaa !4
  %1849 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 44
  %1850 = load float, ptr %1849, align 4, !tbaa !4
  %1851 = load float, ptr %1132, align 4, !tbaa !4
  %1852 = fadd float %1850, %1851
  store float %1852, ptr %1432, align 4, !tbaa !4
  %1853 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 48
  %1854 = load float, ptr %1853, align 4, !tbaa !4
  %1855 = load float, ptr %1133, align 4, !tbaa !4
  %1856 = fadd float %1854, %1855
  store float %1856, ptr %1433, align 16, !tbaa !4
  %1857 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 52
  %1858 = load float, ptr %1857, align 4, !tbaa !4
  %1859 = load float, ptr %1134, align 4, !tbaa !4
  %1860 = fadd float %1858, %1859
  store float %1860, ptr %1434, align 4, !tbaa !4
  %1861 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 56
  %1862 = load float, ptr %1861, align 4, !tbaa !4
  %1863 = load float, ptr %1135, align 4, !tbaa !4
  %1864 = fadd float %1862, %1863
  store float %1864, ptr %1435, align 8, !tbaa !4
  %1865 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 60
  %1866 = load float, ptr %1865, align 4, !tbaa !4
  %1867 = load float, ptr %1136, align 4, !tbaa !4
  %1868 = fadd float %1866, %1867
  store float %1868, ptr %1436, align 4, !tbaa !4
  %1869 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 64
  %1870 = load float, ptr %1869, align 4, !tbaa !4
  %1871 = load float, ptr %1137, align 4, !tbaa !4
  %1872 = fadd float %1870, %1871
  store float %1872, ptr %1437, align 64, !tbaa !4
  %1873 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 68
  %1874 = load float, ptr %1873, align 4, !tbaa !4
  %1875 = load float, ptr %1138, align 4, !tbaa !4
  %1876 = fadd float %1874, %1875
  store float %1876, ptr %1438, align 4, !tbaa !4
  %1877 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 72
  %1878 = load float, ptr %1877, align 4, !tbaa !4
  %1879 = load float, ptr %1139, align 4, !tbaa !4
  %1880 = fadd float %1878, %1879
  store float %1880, ptr %1439, align 8, !tbaa !4
  %1881 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 76
  %1882 = load float, ptr %1881, align 4, !tbaa !4
  %1883 = load float, ptr %1140, align 4, !tbaa !4
  %1884 = fadd float %1882, %1883
  store float %1884, ptr %1440, align 4, !tbaa !4
  %1885 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 80
  %1886 = load float, ptr %1885, align 4, !tbaa !4
  %1887 = load float, ptr %1141, align 4, !tbaa !4
  %1888 = fadd float %1886, %1887
  store float %1888, ptr %1441, align 16, !tbaa !4
  %1889 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 84
  %1890 = load float, ptr %1889, align 4, !tbaa !4
  %1891 = load float, ptr %1142, align 4, !tbaa !4
  %1892 = fadd float %1890, %1891
  store float %1892, ptr %1442, align 4, !tbaa !4
  %1893 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 88
  %1894 = load float, ptr %1893, align 4, !tbaa !4
  %1895 = load float, ptr %1143, align 4, !tbaa !4
  %1896 = fadd float %1894, %1895
  store float %1896, ptr %1443, align 8, !tbaa !4
  %1897 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 92
  %1898 = load float, ptr %1897, align 4, !tbaa !4
  %1899 = load float, ptr %1144, align 4, !tbaa !4
  %1900 = fadd float %1898, %1899
  store float %1900, ptr %1444, align 4, !tbaa !4
  %1901 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 96
  %1902 = load float, ptr %1901, align 4, !tbaa !4
  %1903 = load float, ptr %1145, align 4, !tbaa !4
  %1904 = fadd float %1902, %1903
  store float %1904, ptr %1445, align 32, !tbaa !4
  %1905 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 100
  %1906 = load float, ptr %1905, align 4, !tbaa !4
  %1907 = load float, ptr %1146, align 4, !tbaa !4
  %1908 = fadd float %1906, %1907
  store float %1908, ptr %1446, align 4, !tbaa !4
  %1909 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 104
  %1910 = load float, ptr %1909, align 4, !tbaa !4
  %1911 = load float, ptr %1147, align 4, !tbaa !4
  %1912 = fadd float %1910, %1911
  store float %1912, ptr %1447, align 8, !tbaa !4
  %1913 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 108
  %1914 = load float, ptr %1913, align 4, !tbaa !4
  %1915 = load float, ptr %1148, align 4, !tbaa !4
  %1916 = fadd float %1914, %1915
  store float %1916, ptr %1448, align 4, !tbaa !4
  %1917 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 112
  %1918 = load float, ptr %1917, align 4, !tbaa !4
  %1919 = load float, ptr %1149, align 4, !tbaa !4
  %1920 = fadd float %1918, %1919
  store float %1920, ptr %1449, align 16, !tbaa !4
  %1921 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 116
  %1922 = load float, ptr %1921, align 4, !tbaa !4
  %1923 = load float, ptr %1150, align 4, !tbaa !4
  %1924 = fadd float %1922, %1923
  store float %1924, ptr %1450, align 4, !tbaa !4
  %1925 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 120
  %1926 = load float, ptr %1925, align 4, !tbaa !4
  %1927 = load float, ptr %1151, align 4, !tbaa !4
  %1928 = fadd float %1926, %1927
  store float %1928, ptr %1451, align 8, !tbaa !4
  %1929 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 124
  %1930 = load float, ptr %1929, align 4, !tbaa !4
  %1931 = load float, ptr %1152, align 4, !tbaa !4
  %1932 = fadd float %1930, %1931
  store float %1932, ptr %1452, align 4, !tbaa !4
  %1933 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 128
  %1934 = load float, ptr %1933, align 4, !tbaa !4
  %1935 = load float, ptr %1153, align 4, !tbaa !4
  %1936 = fadd float %1934, %1935
  store float %1936, ptr %1453, align 64, !tbaa !4
  %1937 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 132
  %1938 = load float, ptr %1937, align 4, !tbaa !4
  %1939 = load float, ptr %1154, align 4, !tbaa !4
  %1940 = fadd float %1938, %1939
  store float %1940, ptr %1454, align 4, !tbaa !4
  %1941 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 136
  %1942 = load float, ptr %1941, align 4, !tbaa !4
  %1943 = load float, ptr %1155, align 4, !tbaa !4
  %1944 = fadd float %1942, %1943
  store float %1944, ptr %1455, align 8, !tbaa !4
  %1945 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 140
  %1946 = load float, ptr %1945, align 4, !tbaa !4
  %1947 = load float, ptr %1156, align 4, !tbaa !4
  %1948 = fadd float %1946, %1947
  store float %1948, ptr %1456, align 4, !tbaa !4
  %1949 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 144
  %1950 = load float, ptr %1949, align 4, !tbaa !4
  %1951 = load float, ptr %1157, align 4, !tbaa !4
  %1952 = fadd float %1950, %1951
  store float %1952, ptr %1457, align 16, !tbaa !4
  %1953 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 148
  %1954 = load float, ptr %1953, align 4, !tbaa !4
  %1955 = load float, ptr %1158, align 4, !tbaa !4
  %1956 = fadd float %1954, %1955
  store float %1956, ptr %1458, align 4, !tbaa !4
  %1957 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 152
  %1958 = load float, ptr %1957, align 4, !tbaa !4
  %1959 = load float, ptr %1159, align 4, !tbaa !4
  %1960 = fadd float %1958, %1959
  store float %1960, ptr %1459, align 8, !tbaa !4
  %1961 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 156
  %1962 = load float, ptr %1961, align 4, !tbaa !4
  %1963 = load float, ptr %1160, align 4, !tbaa !4
  %1964 = fadd float %1962, %1963
  store float %1964, ptr %1460, align 4, !tbaa !4
  %1965 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 160
  %1966 = load float, ptr %1965, align 4, !tbaa !4
  %1967 = load float, ptr %1161, align 4, !tbaa !4
  %1968 = fadd float %1966, %1967
  store float %1968, ptr %1461, align 32, !tbaa !4
  %1969 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 164
  %1970 = load float, ptr %1969, align 4, !tbaa !4
  %1971 = load float, ptr %1162, align 4, !tbaa !4
  %1972 = fadd float %1970, %1971
  store float %1972, ptr %1462, align 4, !tbaa !4
  %1973 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 168
  %1974 = load float, ptr %1973, align 4, !tbaa !4
  %1975 = load float, ptr %1163, align 4, !tbaa !4
  %1976 = fadd float %1974, %1975
  store float %1976, ptr %1463, align 8, !tbaa !4
  %1977 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 172
  %1978 = load float, ptr %1977, align 4, !tbaa !4
  %1979 = load float, ptr %1164, align 4, !tbaa !4
  %1980 = fadd float %1978, %1979
  store float %1980, ptr %1464, align 4, !tbaa !4
  %1981 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 176
  %1982 = load float, ptr %1981, align 4, !tbaa !4
  %1983 = load float, ptr %1165, align 4, !tbaa !4
  %1984 = fadd float %1982, %1983
  store float %1984, ptr %1465, align 16, !tbaa !4
  %1985 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 180
  %1986 = load float, ptr %1985, align 4, !tbaa !4
  %1987 = load float, ptr %1166, align 4, !tbaa !4
  %1988 = fadd float %1986, %1987
  store float %1988, ptr %1466, align 4, !tbaa !4
  %1989 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 184
  %1990 = load float, ptr %1989, align 4, !tbaa !4
  %1991 = load float, ptr %1167, align 4, !tbaa !4
  %1992 = fadd float %1990, %1991
  store float %1992, ptr %1467, align 8, !tbaa !4
  %1993 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 188
  %1994 = load float, ptr %1993, align 4, !tbaa !4
  %1995 = load float, ptr %1168, align 4, !tbaa !4
  %1996 = fadd float %1994, %1995
  store float %1996, ptr %1468, align 4, !tbaa !4
  %1997 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 192
  %1998 = load float, ptr %1997, align 4, !tbaa !4
  %1999 = load float, ptr %1169, align 4, !tbaa !4
  %2000 = fadd float %1998, %1999
  store float %2000, ptr %1469, align 64, !tbaa !4
  %2001 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 196
  %2002 = load float, ptr %2001, align 4, !tbaa !4
  %2003 = load float, ptr %1170, align 4, !tbaa !4
  %2004 = fadd float %2002, %2003
  store float %2004, ptr %1470, align 4, !tbaa !4
  %2005 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 200
  %2006 = load float, ptr %2005, align 4, !tbaa !4
  %2007 = load float, ptr %1171, align 4, !tbaa !4
  %2008 = fadd float %2006, %2007
  store float %2008, ptr %1471, align 8, !tbaa !4
  %2009 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 204
  %2010 = load float, ptr %2009, align 4, !tbaa !4
  %2011 = load float, ptr %1172, align 4, !tbaa !4
  %2012 = fadd float %2010, %2011
  store float %2012, ptr %1472, align 4, !tbaa !4
  %2013 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 208
  %2014 = load float, ptr %2013, align 4, !tbaa !4
  %2015 = load float, ptr %1173, align 4, !tbaa !4
  %2016 = fadd float %2014, %2015
  store float %2016, ptr %1473, align 16, !tbaa !4
  %2017 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 212
  %2018 = load float, ptr %2017, align 4, !tbaa !4
  %2019 = load float, ptr %1174, align 4, !tbaa !4
  %2020 = fadd float %2018, %2019
  store float %2020, ptr %1474, align 4, !tbaa !4
  %2021 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 216
  %2022 = load float, ptr %2021, align 4, !tbaa !4
  %2023 = load float, ptr %1175, align 4, !tbaa !4
  %2024 = fadd float %2022, %2023
  store float %2024, ptr %1475, align 8, !tbaa !4
  %2025 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 220
  %2026 = load float, ptr %2025, align 4, !tbaa !4
  %2027 = load float, ptr %1176, align 4, !tbaa !4
  %2028 = fadd float %2026, %2027
  store float %2028, ptr %1476, align 4, !tbaa !4
  %2029 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 224
  %2030 = load float, ptr %2029, align 4, !tbaa !4
  %2031 = load float, ptr %1177, align 4, !tbaa !4
  %2032 = fadd float %2030, %2031
  store float %2032, ptr %1477, align 32, !tbaa !4
  %2033 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 228
  %2034 = load float, ptr %2033, align 4, !tbaa !4
  %2035 = load float, ptr %1178, align 4, !tbaa !4
  %2036 = fadd float %2034, %2035
  store float %2036, ptr %1478, align 4, !tbaa !4
  %2037 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 232
  %2038 = load float, ptr %2037, align 4, !tbaa !4
  %2039 = load float, ptr %1179, align 4, !tbaa !4
  %2040 = fadd float %2038, %2039
  store float %2040, ptr %1479, align 8, !tbaa !4
  %2041 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 236
  %2042 = load float, ptr %2041, align 4, !tbaa !4
  %2043 = load float, ptr %1180, align 4, !tbaa !4
  %2044 = fadd float %2042, %2043
  store float %2044, ptr %1480, align 4, !tbaa !4
  %2045 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 240
  %2046 = load float, ptr %2045, align 4, !tbaa !4
  %2047 = load float, ptr %1181, align 4, !tbaa !4
  %2048 = fadd float %2046, %2047
  store float %2048, ptr %1481, align 16, !tbaa !4
  %2049 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 244
  %2050 = load float, ptr %2049, align 4, !tbaa !4
  %2051 = load float, ptr %1182, align 4, !tbaa !4
  %2052 = fadd float %2050, %2051
  store float %2052, ptr %1482, align 4, !tbaa !4
  %2053 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 248
  %2054 = load float, ptr %2053, align 4, !tbaa !4
  %2055 = load float, ptr %1183, align 4, !tbaa !4
  %2056 = fadd float %2054, %2055
  store float %2056, ptr %1483, align 8, !tbaa !4
  %2057 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 252
  %2058 = load float, ptr %2057, align 4, !tbaa !4
  %2059 = load float, ptr %1184, align 4, !tbaa !4
  %2060 = fadd float %2058, %2059
  store float %2060, ptr %1484, align 4, !tbaa !4
  %2061 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 256
  %2062 = load float, ptr %2061, align 4, !tbaa !4
  %2063 = load float, ptr %1185, align 4, !tbaa !4
  %2064 = fadd float %2062, %2063
  store float %2064, ptr %1485, align 64, !tbaa !4
  %2065 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 260
  %2066 = load float, ptr %2065, align 4, !tbaa !4
  %2067 = load float, ptr %1186, align 4, !tbaa !4
  %2068 = fadd float %2066, %2067
  store float %2068, ptr %1486, align 4, !tbaa !4
  %2069 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 264
  %2070 = load float, ptr %2069, align 4, !tbaa !4
  %2071 = load float, ptr %1187, align 4, !tbaa !4
  %2072 = fadd float %2070, %2071
  store float %2072, ptr %1487, align 8, !tbaa !4
  %2073 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 268
  %2074 = load float, ptr %2073, align 4, !tbaa !4
  %2075 = load float, ptr %1188, align 4, !tbaa !4
  %2076 = fadd float %2074, %2075
  store float %2076, ptr %1488, align 4, !tbaa !4
  %2077 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 272
  %2078 = load float, ptr %2077, align 4, !tbaa !4
  %2079 = load float, ptr %1189, align 4, !tbaa !4
  %2080 = fadd float %2078, %2079
  store float %2080, ptr %1489, align 16, !tbaa !4
  %2081 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 276
  %2082 = load float, ptr %2081, align 4, !tbaa !4
  %2083 = load float, ptr %1190, align 4, !tbaa !4
  %2084 = fadd float %2082, %2083
  store float %2084, ptr %1490, align 4, !tbaa !4
  %2085 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 280
  %2086 = load float, ptr %2085, align 4, !tbaa !4
  %2087 = load float, ptr %1191, align 4, !tbaa !4
  %2088 = fadd float %2086, %2087
  store float %2088, ptr %1491, align 8, !tbaa !4
  %2089 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 284
  %2090 = load float, ptr %2089, align 4, !tbaa !4
  %2091 = load float, ptr %1192, align 4, !tbaa !4
  %2092 = fadd float %2090, %2091
  store float %2092, ptr %1492, align 4, !tbaa !4
  %2093 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 288
  %2094 = load float, ptr %2093, align 4, !tbaa !4
  %2095 = load float, ptr %1193, align 4, !tbaa !4
  %2096 = fadd float %2094, %2095
  store float %2096, ptr %1493, align 32, !tbaa !4
  %2097 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 292
  %2098 = load float, ptr %2097, align 4, !tbaa !4
  %2099 = load float, ptr %1194, align 4, !tbaa !4
  %2100 = fadd float %2098, %2099
  store float %2100, ptr %1494, align 4, !tbaa !4
  %2101 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 296
  %2102 = load float, ptr %2101, align 4, !tbaa !4
  %2103 = load float, ptr %1195, align 4, !tbaa !4
  %2104 = fadd float %2102, %2103
  store float %2104, ptr %1495, align 8, !tbaa !4
  %2105 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 300
  %2106 = load float, ptr %2105, align 4, !tbaa !4
  %2107 = load float, ptr %1196, align 4, !tbaa !4
  %2108 = fadd float %2106, %2107
  store float %2108, ptr %1496, align 4, !tbaa !4
  %2109 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 304
  %2110 = load float, ptr %2109, align 4, !tbaa !4
  %2111 = load float, ptr %1197, align 4, !tbaa !4
  %2112 = fadd float %2110, %2111
  store float %2112, ptr %1497, align 16, !tbaa !4
  %2113 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 308
  %2114 = load float, ptr %2113, align 4, !tbaa !4
  %2115 = load float, ptr %1198, align 4, !tbaa !4
  %2116 = fadd float %2114, %2115
  store float %2116, ptr %1498, align 4, !tbaa !4
  %2117 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 312
  %2118 = load float, ptr %2117, align 4, !tbaa !4
  %2119 = load float, ptr %1199, align 4, !tbaa !4
  %2120 = fadd float %2118, %2119
  store float %2120, ptr %1499, align 8, !tbaa !4
  %2121 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 316
  %2122 = load float, ptr %2121, align 4, !tbaa !4
  %2123 = load float, ptr %1200, align 4, !tbaa !4
  %2124 = fadd float %2122, %2123
  store float %2124, ptr %1500, align 4, !tbaa !4
  %2125 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 320
  %2126 = load float, ptr %2125, align 4, !tbaa !4
  %2127 = load float, ptr %1201, align 4, !tbaa !4
  %2128 = fadd float %2126, %2127
  store float %2128, ptr %1501, align 64, !tbaa !4
  %2129 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 324
  %2130 = load float, ptr %2129, align 4, !tbaa !4
  %2131 = load float, ptr %1202, align 4, !tbaa !4
  %2132 = fadd float %2130, %2131
  store float %2132, ptr %1502, align 4, !tbaa !4
  %2133 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 328
  %2134 = load float, ptr %2133, align 4, !tbaa !4
  %2135 = load float, ptr %1203, align 4, !tbaa !4
  %2136 = fadd float %2134, %2135
  store float %2136, ptr %1503, align 8, !tbaa !4
  %2137 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 332
  %2138 = load float, ptr %2137, align 4, !tbaa !4
  %2139 = load float, ptr %1204, align 4, !tbaa !4
  %2140 = fadd float %2138, %2139
  store float %2140, ptr %1504, align 4, !tbaa !4
  %2141 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 336
  %2142 = load float, ptr %2141, align 4, !tbaa !4
  %2143 = load float, ptr %1205, align 4, !tbaa !4
  %2144 = fadd float %2142, %2143
  store float %2144, ptr %1505, align 16, !tbaa !4
  %2145 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 340
  %2146 = load float, ptr %2145, align 4, !tbaa !4
  %2147 = load float, ptr %1206, align 4, !tbaa !4
  %2148 = fadd float %2146, %2147
  store float %2148, ptr %1506, align 4, !tbaa !4
  %2149 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 344
  %2150 = load float, ptr %2149, align 4, !tbaa !4
  %2151 = load float, ptr %1207, align 4, !tbaa !4
  %2152 = fadd float %2150, %2151
  store float %2152, ptr %1507, align 8, !tbaa !4
  %2153 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 348
  %2154 = load float, ptr %2153, align 4, !tbaa !4
  %2155 = load float, ptr %1208, align 4, !tbaa !4
  %2156 = fadd float %2154, %2155
  store float %2156, ptr %1508, align 4, !tbaa !4
  %2157 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 352
  %2158 = load float, ptr %2157, align 4, !tbaa !4
  %2159 = load float, ptr %1209, align 4, !tbaa !4
  %2160 = fadd float %2158, %2159
  store float %2160, ptr %1509, align 32, !tbaa !4
  %2161 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 356
  %2162 = load float, ptr %2161, align 4, !tbaa !4
  %2163 = load float, ptr %1210, align 4, !tbaa !4
  %2164 = fadd float %2162, %2163
  store float %2164, ptr %1510, align 4, !tbaa !4
  %2165 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 360
  %2166 = load float, ptr %2165, align 4, !tbaa !4
  %2167 = load float, ptr %1211, align 4, !tbaa !4
  %2168 = fadd float %2166, %2167
  store float %2168, ptr %1511, align 8, !tbaa !4
  %2169 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 364
  %2170 = load float, ptr %2169, align 4, !tbaa !4
  %2171 = load float, ptr %1212, align 4, !tbaa !4
  %2172 = fadd float %2170, %2171
  store float %2172, ptr %1512, align 4, !tbaa !4
  %2173 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 368
  %2174 = load float, ptr %2173, align 4, !tbaa !4
  %2175 = load float, ptr %1213, align 4, !tbaa !4
  %2176 = fadd float %2174, %2175
  store float %2176, ptr %1513, align 16, !tbaa !4
  %2177 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 372
  %2178 = load float, ptr %2177, align 4, !tbaa !4
  %2179 = load float, ptr %1214, align 4, !tbaa !4
  %2180 = fadd float %2178, %2179
  store float %2180, ptr %1514, align 4, !tbaa !4
  %2181 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 376
  %2182 = load float, ptr %2181, align 4, !tbaa !4
  %2183 = load float, ptr %1215, align 4, !tbaa !4
  %2184 = fadd float %2182, %2183
  store float %2184, ptr %1515, align 8, !tbaa !4
  %2185 = getelementptr inbounds nuw i8, ptr %.pn297418, i64 380
  %2186 = load float, ptr %2185, align 4, !tbaa !4
  %2187 = load float, ptr %1216, align 4, !tbaa !4
  %2188 = fadd float %2186, %2187
  store float %2188, ptr %1516, align 4, !tbaa !4
  %2189 = fmul double %1803, %1420
  %2190 = fadd double %2189, %1421
  %2191 = fpext float %1525 to double
  %2192 = fpext float %1527 to double
  %.inv = fcmp ole double %2190, 0.000000e+00
  %2193 = select i1 %.inv, double 0.000000e+00, double %2190
  %2194 = fcmp uno double %2190, 0.000000e+00
  %2195 = tail call double @llvm.fabs.f64(double %2190)
  %2196 = fneg double %2195
  %2197 = tail call double @llvm.exp.f64(double %2196)
  %2198 = fadd double %2197, 1.000000e+00
  %2199 = tail call double @llvm.log.f64(double %2198)
  %2200 = fadd double %2193, %2199
  %2201 = select i1 %2194, double %2190, double %2200
  %2202 = fmul double %2190, %2191
  %2203 = fsub double %2201, %2202
  %2204 = fmul double %2203, %2192
  tail call void @_mlir_memref_to_llvm_free(ptr nonnull %.pn297418)
  %2205 = fpext float %1805 to double
  %2206 = fadd double %1519, %2205
  %2207 = fpext float %1804 to double
  %2208 = fadd double %1518, %2207
  %2209 = fadd double %1517, %2204
  %2210 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 448)
  %2211 = ptrtoint ptr %2210 to i64
  %2212 = add i64 %2211, 63
  %2213 = and i64 %2212, -64
  %2214 = inttoptr i64 %2213 to ptr
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 64 dereferenceable(384) %2214, ptr noundef nonnull align 64 dereferenceable(384) %1321, i64 384, i1 false)
  %2215 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 384)
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(384) %2215, ptr noundef nonnull align 64 dereferenceable(384) %2214, i64 384, i1 false)
  tail call void @_mlir_memref_to_llvm_free(ptr %2210)
  %2216 = add nuw nsw i64 %1520, 1
  %exitcond460.not = icmp eq i64 %2216, 32
  br i1 %exitcond460.not, label %.preheader338, label %.preheader332

.preheader338:                                    ; preds = %.preheader332, %.preheader338
  %2217 = phi i64 [ %2242, %.preheader338 ], [ 0, %.preheader332 ]
  %.idx301 = mul nuw nsw i64 %2217, 96
  %2218 = getelementptr i8, ptr %1321, i64 %.idx301
  store float 3.125000e-02, ptr %2218, align 32, !tbaa !4
  %2219 = getelementptr i8, ptr %2218, i64 4
  store float 3.125000e-02, ptr %2219, align 4, !tbaa !4
  %2220 = getelementptr i8, ptr %2218, i64 8
  store float 3.125000e-02, ptr %2220, align 8, !tbaa !4
  %2221 = getelementptr i8, ptr %2218, i64 12
  store float 3.125000e-02, ptr %2221, align 4, !tbaa !4
  %2222 = getelementptr i8, ptr %2218, i64 16
  store float 3.125000e-02, ptr %2222, align 16, !tbaa !4
  %2223 = getelementptr i8, ptr %2218, i64 20
  store float 3.125000e-02, ptr %2223, align 4, !tbaa !4
  %2224 = getelementptr i8, ptr %2218, i64 24
  store float 3.125000e-02, ptr %2224, align 8, !tbaa !4
  %2225 = getelementptr i8, ptr %2218, i64 28
  store float 3.125000e-02, ptr %2225, align 4, !tbaa !4
  %2226 = getelementptr i8, ptr %2218, i64 32
  store float 3.125000e-02, ptr %2226, align 32, !tbaa !4
  %2227 = getelementptr i8, ptr %2218, i64 36
  store float 3.125000e-02, ptr %2227, align 4, !tbaa !4
  %2228 = getelementptr i8, ptr %2218, i64 40
  store float 3.125000e-02, ptr %2228, align 8, !tbaa !4
  %2229 = getelementptr i8, ptr %2218, i64 44
  store float 3.125000e-02, ptr %2229, align 4, !tbaa !4
  %2230 = getelementptr i8, ptr %2218, i64 48
  store float 3.125000e-02, ptr %2230, align 16, !tbaa !4
  %2231 = getelementptr i8, ptr %2218, i64 52
  store float 3.125000e-02, ptr %2231, align 4, !tbaa !4
  %2232 = getelementptr i8, ptr %2218, i64 56
  store float 3.125000e-02, ptr %2232, align 8, !tbaa !4
  %2233 = getelementptr i8, ptr %2218, i64 60
  store float 3.125000e-02, ptr %2233, align 4, !tbaa !4
  %2234 = getelementptr i8, ptr %2218, i64 64
  store float 3.125000e-02, ptr %2234, align 32, !tbaa !4
  %2235 = getelementptr i8, ptr %2218, i64 68
  store float 3.125000e-02, ptr %2235, align 4, !tbaa !4
  %2236 = getelementptr i8, ptr %2218, i64 72
  store float 3.125000e-02, ptr %2236, align 8, !tbaa !4
  %2237 = getelementptr i8, ptr %2218, i64 76
  store float 3.125000e-02, ptr %2237, align 4, !tbaa !4
  %2238 = getelementptr i8, ptr %2218, i64 80
  store float 3.125000e-02, ptr %2238, align 16, !tbaa !4
  %2239 = getelementptr i8, ptr %2218, i64 84
  store float 3.125000e-02, ptr %2239, align 4, !tbaa !4
  %2240 = getelementptr i8, ptr %2218, i64 88
  store float 3.125000e-02, ptr %2240, align 8, !tbaa !4
  %2241 = getelementptr i8, ptr %2218, i64 92
  store float 3.125000e-02, ptr %2241, align 4, !tbaa !4
  %2242 = add nuw nsw i64 %2217, 1
  %exitcond463.not = icmp eq i64 %2242, 4
  br i1 %exitcond463.not, label %.preheader337.preheader, label %.preheader338

.preheader337.preheader:                          ; preds = %.preheader338
  %2243 = load float, ptr %2215, align 4, !tbaa !4
  %2244 = load float, ptr %1321, align 64, !tbaa !4
  %2245 = fmul float %2243, %2244
  store float %2245, ptr %1109, align 64, !tbaa !4
  %2246 = getelementptr inbounds nuw i8, ptr %2215, i64 4
  %2247 = load float, ptr %2246, align 4, !tbaa !4
  %2248 = load float, ptr %1422, align 4, !tbaa !4
  %2249 = fmul float %2247, %2248
  store float %2249, ptr %1217, align 4, !tbaa !4
  %2250 = getelementptr inbounds nuw i8, ptr %2215, i64 8
  %2251 = load float, ptr %2250, align 4, !tbaa !4
  %2252 = load float, ptr %1423, align 8, !tbaa !4
  %2253 = fmul float %2251, %2252
  store float %2253, ptr %1218, align 8, !tbaa !4
  %2254 = getelementptr inbounds nuw i8, ptr %2215, i64 12
  %2255 = load float, ptr %2254, align 4, !tbaa !4
  %2256 = load float, ptr %1424, align 4, !tbaa !4
  %2257 = fmul float %2255, %2256
  store float %2257, ptr %1219, align 4, !tbaa !4
  %2258 = getelementptr inbounds nuw i8, ptr %2215, i64 16
  %2259 = load float, ptr %2258, align 4, !tbaa !4
  %2260 = load float, ptr %1425, align 16, !tbaa !4
  %2261 = fmul float %2259, %2260
  store float %2261, ptr %1220, align 16, !tbaa !4
  %2262 = getelementptr inbounds nuw i8, ptr %2215, i64 20
  %2263 = load float, ptr %2262, align 4, !tbaa !4
  %2264 = load float, ptr %1426, align 4, !tbaa !4
  %2265 = fmul float %2263, %2264
  store float %2265, ptr %1221, align 4, !tbaa !4
  %2266 = getelementptr inbounds nuw i8, ptr %2215, i64 24
  %2267 = load float, ptr %2266, align 4, !tbaa !4
  %2268 = load float, ptr %1427, align 8, !tbaa !4
  %2269 = fmul float %2267, %2268
  store float %2269, ptr %1222, align 8, !tbaa !4
  %2270 = getelementptr inbounds nuw i8, ptr %2215, i64 28
  %2271 = load float, ptr %2270, align 4, !tbaa !4
  %2272 = load float, ptr %1428, align 4, !tbaa !4
  %2273 = fmul float %2271, %2272
  store float %2273, ptr %1223, align 4, !tbaa !4
  %2274 = getelementptr inbounds nuw i8, ptr %2215, i64 32
  %2275 = load float, ptr %2274, align 4, !tbaa !4
  %2276 = load float, ptr %1429, align 32, !tbaa !4
  %2277 = fmul float %2275, %2276
  store float %2277, ptr %1224, align 32, !tbaa !4
  %2278 = getelementptr inbounds nuw i8, ptr %2215, i64 36
  %2279 = load float, ptr %2278, align 4, !tbaa !4
  %2280 = load float, ptr %1430, align 4, !tbaa !4
  %2281 = fmul float %2279, %2280
  store float %2281, ptr %1225, align 4, !tbaa !4
  %2282 = getelementptr inbounds nuw i8, ptr %2215, i64 40
  %2283 = load float, ptr %2282, align 4, !tbaa !4
  %2284 = load float, ptr %1431, align 8, !tbaa !4
  %2285 = fmul float %2283, %2284
  store float %2285, ptr %1226, align 8, !tbaa !4
  %2286 = getelementptr inbounds nuw i8, ptr %2215, i64 44
  %2287 = load float, ptr %2286, align 4, !tbaa !4
  %2288 = load float, ptr %1432, align 4, !tbaa !4
  %2289 = fmul float %2287, %2288
  store float %2289, ptr %1227, align 4, !tbaa !4
  %2290 = getelementptr inbounds nuw i8, ptr %2215, i64 48
  %2291 = load float, ptr %2290, align 4, !tbaa !4
  %2292 = load float, ptr %1433, align 16, !tbaa !4
  %2293 = fmul float %2291, %2292
  store float %2293, ptr %1228, align 16, !tbaa !4
  %2294 = getelementptr inbounds nuw i8, ptr %2215, i64 52
  %2295 = load float, ptr %2294, align 4, !tbaa !4
  %2296 = load float, ptr %1434, align 4, !tbaa !4
  %2297 = fmul float %2295, %2296
  store float %2297, ptr %1229, align 4, !tbaa !4
  %2298 = getelementptr inbounds nuw i8, ptr %2215, i64 56
  %2299 = load float, ptr %2298, align 4, !tbaa !4
  %2300 = load float, ptr %1435, align 8, !tbaa !4
  %2301 = fmul float %2299, %2300
  store float %2301, ptr %1230, align 8, !tbaa !4
  %2302 = getelementptr inbounds nuw i8, ptr %2215, i64 60
  %2303 = load float, ptr %2302, align 4, !tbaa !4
  %2304 = load float, ptr %1436, align 4, !tbaa !4
  %2305 = fmul float %2303, %2304
  store float %2305, ptr %1231, align 4, !tbaa !4
  %2306 = getelementptr inbounds nuw i8, ptr %2215, i64 64
  %2307 = load float, ptr %2306, align 4, !tbaa !4
  %2308 = load float, ptr %1437, align 64, !tbaa !4
  %2309 = fmul float %2307, %2308
  store float %2309, ptr %1232, align 64, !tbaa !4
  %2310 = getelementptr inbounds nuw i8, ptr %2215, i64 68
  %2311 = load float, ptr %2310, align 4, !tbaa !4
  %2312 = load float, ptr %1438, align 4, !tbaa !4
  %2313 = fmul float %2311, %2312
  store float %2313, ptr %1233, align 4, !tbaa !4
  %2314 = getelementptr inbounds nuw i8, ptr %2215, i64 72
  %2315 = load float, ptr %2314, align 4, !tbaa !4
  %2316 = load float, ptr %1439, align 8, !tbaa !4
  %2317 = fmul float %2315, %2316
  store float %2317, ptr %1234, align 8, !tbaa !4
  %2318 = getelementptr inbounds nuw i8, ptr %2215, i64 76
  %2319 = load float, ptr %2318, align 4, !tbaa !4
  %2320 = load float, ptr %1440, align 4, !tbaa !4
  %2321 = fmul float %2319, %2320
  store float %2321, ptr %1235, align 4, !tbaa !4
  %2322 = getelementptr inbounds nuw i8, ptr %2215, i64 80
  %2323 = load float, ptr %2322, align 4, !tbaa !4
  %2324 = load float, ptr %1441, align 16, !tbaa !4
  %2325 = fmul float %2323, %2324
  store float %2325, ptr %1236, align 16, !tbaa !4
  %2326 = getelementptr inbounds nuw i8, ptr %2215, i64 84
  %2327 = load float, ptr %2326, align 4, !tbaa !4
  %2328 = load float, ptr %1442, align 4, !tbaa !4
  %2329 = fmul float %2327, %2328
  store float %2329, ptr %1237, align 4, !tbaa !4
  %2330 = getelementptr inbounds nuw i8, ptr %2215, i64 88
  %2331 = load float, ptr %2330, align 4, !tbaa !4
  %2332 = load float, ptr %1443, align 8, !tbaa !4
  %2333 = fmul float %2331, %2332
  store float %2333, ptr %1238, align 8, !tbaa !4
  %2334 = getelementptr inbounds nuw i8, ptr %2215, i64 92
  %2335 = load float, ptr %2334, align 4, !tbaa !4
  %2336 = load float, ptr %1444, align 4, !tbaa !4
  %2337 = fmul float %2335, %2336
  store float %2337, ptr %1239, align 4, !tbaa !4
  %2338 = getelementptr inbounds nuw i8, ptr %2215, i64 96
  %2339 = load float, ptr %2338, align 4, !tbaa !4
  %2340 = load float, ptr %1445, align 32, !tbaa !4
  %2341 = fmul float %2339, %2340
  store float %2341, ptr %1240, align 32, !tbaa !4
  %2342 = getelementptr inbounds nuw i8, ptr %2215, i64 100
  %2343 = load float, ptr %2342, align 4, !tbaa !4
  %2344 = load float, ptr %1446, align 4, !tbaa !4
  %2345 = fmul float %2343, %2344
  store float %2345, ptr %1241, align 4, !tbaa !4
  %2346 = getelementptr inbounds nuw i8, ptr %2215, i64 104
  %2347 = load float, ptr %2346, align 4, !tbaa !4
  %2348 = load float, ptr %1447, align 8, !tbaa !4
  %2349 = fmul float %2347, %2348
  store float %2349, ptr %1242, align 8, !tbaa !4
  %2350 = getelementptr inbounds nuw i8, ptr %2215, i64 108
  %2351 = load float, ptr %2350, align 4, !tbaa !4
  %2352 = load float, ptr %1448, align 4, !tbaa !4
  %2353 = fmul float %2351, %2352
  store float %2353, ptr %1243, align 4, !tbaa !4
  %2354 = getelementptr inbounds nuw i8, ptr %2215, i64 112
  %2355 = load float, ptr %2354, align 4, !tbaa !4
  %2356 = load float, ptr %1449, align 16, !tbaa !4
  %2357 = fmul float %2355, %2356
  store float %2357, ptr %1244, align 16, !tbaa !4
  %2358 = getelementptr inbounds nuw i8, ptr %2215, i64 116
  %2359 = load float, ptr %2358, align 4, !tbaa !4
  %2360 = load float, ptr %1450, align 4, !tbaa !4
  %2361 = fmul float %2359, %2360
  store float %2361, ptr %1245, align 4, !tbaa !4
  %2362 = getelementptr inbounds nuw i8, ptr %2215, i64 120
  %2363 = load float, ptr %2362, align 4, !tbaa !4
  %2364 = load float, ptr %1451, align 8, !tbaa !4
  %2365 = fmul float %2363, %2364
  store float %2365, ptr %1246, align 8, !tbaa !4
  %2366 = getelementptr inbounds nuw i8, ptr %2215, i64 124
  %2367 = load float, ptr %2366, align 4, !tbaa !4
  %2368 = load float, ptr %1452, align 4, !tbaa !4
  %2369 = fmul float %2367, %2368
  store float %2369, ptr %1247, align 4, !tbaa !4
  %2370 = getelementptr inbounds nuw i8, ptr %2215, i64 128
  %2371 = load float, ptr %2370, align 4, !tbaa !4
  %2372 = load float, ptr %1453, align 64, !tbaa !4
  %2373 = fmul float %2371, %2372
  store float %2373, ptr %1248, align 64, !tbaa !4
  %2374 = getelementptr inbounds nuw i8, ptr %2215, i64 132
  %2375 = load float, ptr %2374, align 4, !tbaa !4
  %2376 = load float, ptr %1454, align 4, !tbaa !4
  %2377 = fmul float %2375, %2376
  store float %2377, ptr %1249, align 4, !tbaa !4
  %2378 = getelementptr inbounds nuw i8, ptr %2215, i64 136
  %2379 = load float, ptr %2378, align 4, !tbaa !4
  %2380 = load float, ptr %1455, align 8, !tbaa !4
  %2381 = fmul float %2379, %2380
  store float %2381, ptr %1250, align 8, !tbaa !4
  %2382 = getelementptr inbounds nuw i8, ptr %2215, i64 140
  %2383 = load float, ptr %2382, align 4, !tbaa !4
  %2384 = load float, ptr %1456, align 4, !tbaa !4
  %2385 = fmul float %2383, %2384
  store float %2385, ptr %1251, align 4, !tbaa !4
  %2386 = getelementptr inbounds nuw i8, ptr %2215, i64 144
  %2387 = load float, ptr %2386, align 4, !tbaa !4
  %2388 = load float, ptr %1457, align 16, !tbaa !4
  %2389 = fmul float %2387, %2388
  store float %2389, ptr %1252, align 16, !tbaa !4
  %2390 = getelementptr inbounds nuw i8, ptr %2215, i64 148
  %2391 = load float, ptr %2390, align 4, !tbaa !4
  %2392 = load float, ptr %1458, align 4, !tbaa !4
  %2393 = fmul float %2391, %2392
  store float %2393, ptr %1253, align 4, !tbaa !4
  %2394 = getelementptr inbounds nuw i8, ptr %2215, i64 152
  %2395 = load float, ptr %2394, align 4, !tbaa !4
  %2396 = load float, ptr %1459, align 8, !tbaa !4
  %2397 = fmul float %2395, %2396
  store float %2397, ptr %1254, align 8, !tbaa !4
  %2398 = getelementptr inbounds nuw i8, ptr %2215, i64 156
  %2399 = load float, ptr %2398, align 4, !tbaa !4
  %2400 = load float, ptr %1460, align 4, !tbaa !4
  %2401 = fmul float %2399, %2400
  store float %2401, ptr %1255, align 4, !tbaa !4
  %2402 = getelementptr inbounds nuw i8, ptr %2215, i64 160
  %2403 = load float, ptr %2402, align 4, !tbaa !4
  %2404 = load float, ptr %1461, align 32, !tbaa !4
  %2405 = fmul float %2403, %2404
  store float %2405, ptr %1256, align 32, !tbaa !4
  %2406 = getelementptr inbounds nuw i8, ptr %2215, i64 164
  %2407 = load float, ptr %2406, align 4, !tbaa !4
  %2408 = load float, ptr %1462, align 4, !tbaa !4
  %2409 = fmul float %2407, %2408
  store float %2409, ptr %1257, align 4, !tbaa !4
  %2410 = getelementptr inbounds nuw i8, ptr %2215, i64 168
  %2411 = load float, ptr %2410, align 4, !tbaa !4
  %2412 = load float, ptr %1463, align 8, !tbaa !4
  %2413 = fmul float %2411, %2412
  store float %2413, ptr %1258, align 8, !tbaa !4
  %2414 = getelementptr inbounds nuw i8, ptr %2215, i64 172
  %2415 = load float, ptr %2414, align 4, !tbaa !4
  %2416 = load float, ptr %1464, align 4, !tbaa !4
  %2417 = fmul float %2415, %2416
  store float %2417, ptr %1259, align 4, !tbaa !4
  %2418 = getelementptr inbounds nuw i8, ptr %2215, i64 176
  %2419 = load float, ptr %2418, align 4, !tbaa !4
  %2420 = load float, ptr %1465, align 16, !tbaa !4
  %2421 = fmul float %2419, %2420
  store float %2421, ptr %1260, align 16, !tbaa !4
  %2422 = getelementptr inbounds nuw i8, ptr %2215, i64 180
  %2423 = load float, ptr %2422, align 4, !tbaa !4
  %2424 = load float, ptr %1466, align 4, !tbaa !4
  %2425 = fmul float %2423, %2424
  store float %2425, ptr %1261, align 4, !tbaa !4
  %2426 = getelementptr inbounds nuw i8, ptr %2215, i64 184
  %2427 = load float, ptr %2426, align 4, !tbaa !4
  %2428 = load float, ptr %1467, align 8, !tbaa !4
  %2429 = fmul float %2427, %2428
  store float %2429, ptr %1262, align 8, !tbaa !4
  %2430 = getelementptr inbounds nuw i8, ptr %2215, i64 188
  %2431 = load float, ptr %2430, align 4, !tbaa !4
  %2432 = load float, ptr %1468, align 4, !tbaa !4
  %2433 = fmul float %2431, %2432
  store float %2433, ptr %1263, align 4, !tbaa !4
  %2434 = getelementptr inbounds nuw i8, ptr %2215, i64 192
  %2435 = load float, ptr %2434, align 4, !tbaa !4
  %2436 = load float, ptr %1469, align 64, !tbaa !4
  %2437 = fmul float %2435, %2436
  store float %2437, ptr %1264, align 64, !tbaa !4
  %2438 = getelementptr inbounds nuw i8, ptr %2215, i64 196
  %2439 = load float, ptr %2438, align 4, !tbaa !4
  %2440 = load float, ptr %1470, align 4, !tbaa !4
  %2441 = fmul float %2439, %2440
  store float %2441, ptr %1265, align 4, !tbaa !4
  %2442 = getelementptr inbounds nuw i8, ptr %2215, i64 200
  %2443 = load float, ptr %2442, align 4, !tbaa !4
  %2444 = load float, ptr %1471, align 8, !tbaa !4
  %2445 = fmul float %2443, %2444
  store float %2445, ptr %1266, align 8, !tbaa !4
  %2446 = getelementptr inbounds nuw i8, ptr %2215, i64 204
  %2447 = load float, ptr %2446, align 4, !tbaa !4
  %2448 = load float, ptr %1472, align 4, !tbaa !4
  %2449 = fmul float %2447, %2448
  store float %2449, ptr %1267, align 4, !tbaa !4
  %2450 = getelementptr inbounds nuw i8, ptr %2215, i64 208
  %2451 = load float, ptr %2450, align 4, !tbaa !4
  %2452 = load float, ptr %1473, align 16, !tbaa !4
  %2453 = fmul float %2451, %2452
  store float %2453, ptr %1268, align 16, !tbaa !4
  %2454 = getelementptr inbounds nuw i8, ptr %2215, i64 212
  %2455 = load float, ptr %2454, align 4, !tbaa !4
  %2456 = load float, ptr %1474, align 4, !tbaa !4
  %2457 = fmul float %2455, %2456
  store float %2457, ptr %1269, align 4, !tbaa !4
  %2458 = getelementptr inbounds nuw i8, ptr %2215, i64 216
  %2459 = load float, ptr %2458, align 4, !tbaa !4
  %2460 = load float, ptr %1475, align 8, !tbaa !4
  %2461 = fmul float %2459, %2460
  store float %2461, ptr %1270, align 8, !tbaa !4
  %2462 = getelementptr inbounds nuw i8, ptr %2215, i64 220
  %2463 = load float, ptr %2462, align 4, !tbaa !4
  %2464 = load float, ptr %1476, align 4, !tbaa !4
  %2465 = fmul float %2463, %2464
  store float %2465, ptr %1271, align 4, !tbaa !4
  %2466 = getelementptr inbounds nuw i8, ptr %2215, i64 224
  %2467 = load float, ptr %2466, align 4, !tbaa !4
  %2468 = load float, ptr %1477, align 32, !tbaa !4
  %2469 = fmul float %2467, %2468
  store float %2469, ptr %1272, align 32, !tbaa !4
  %2470 = getelementptr inbounds nuw i8, ptr %2215, i64 228
  %2471 = load float, ptr %2470, align 4, !tbaa !4
  %2472 = load float, ptr %1478, align 4, !tbaa !4
  %2473 = fmul float %2471, %2472
  store float %2473, ptr %1273, align 4, !tbaa !4
  %2474 = getelementptr inbounds nuw i8, ptr %2215, i64 232
  %2475 = load float, ptr %2474, align 4, !tbaa !4
  %2476 = load float, ptr %1479, align 8, !tbaa !4
  %2477 = fmul float %2475, %2476
  store float %2477, ptr %1274, align 8, !tbaa !4
  %2478 = getelementptr inbounds nuw i8, ptr %2215, i64 236
  %2479 = load float, ptr %2478, align 4, !tbaa !4
  %2480 = load float, ptr %1480, align 4, !tbaa !4
  %2481 = fmul float %2479, %2480
  store float %2481, ptr %1275, align 4, !tbaa !4
  %2482 = getelementptr inbounds nuw i8, ptr %2215, i64 240
  %2483 = load float, ptr %2482, align 4, !tbaa !4
  %2484 = load float, ptr %1481, align 16, !tbaa !4
  %2485 = fmul float %2483, %2484
  store float %2485, ptr %1276, align 16, !tbaa !4
  %2486 = getelementptr inbounds nuw i8, ptr %2215, i64 244
  %2487 = load float, ptr %2486, align 4, !tbaa !4
  %2488 = load float, ptr %1482, align 4, !tbaa !4
  %2489 = fmul float %2487, %2488
  store float %2489, ptr %1277, align 4, !tbaa !4
  %2490 = getelementptr inbounds nuw i8, ptr %2215, i64 248
  %2491 = load float, ptr %2490, align 4, !tbaa !4
  %2492 = load float, ptr %1483, align 8, !tbaa !4
  %2493 = fmul float %2491, %2492
  store float %2493, ptr %1278, align 8, !tbaa !4
  %2494 = getelementptr inbounds nuw i8, ptr %2215, i64 252
  %2495 = load float, ptr %2494, align 4, !tbaa !4
  %2496 = load float, ptr %1484, align 4, !tbaa !4
  %2497 = fmul float %2495, %2496
  store float %2497, ptr %1279, align 4, !tbaa !4
  %2498 = getelementptr inbounds nuw i8, ptr %2215, i64 256
  %2499 = load float, ptr %2498, align 4, !tbaa !4
  %2500 = load float, ptr %1485, align 64, !tbaa !4
  %2501 = fmul float %2499, %2500
  store float %2501, ptr %1280, align 64, !tbaa !4
  %2502 = getelementptr inbounds nuw i8, ptr %2215, i64 260
  %2503 = load float, ptr %2502, align 4, !tbaa !4
  %2504 = load float, ptr %1486, align 4, !tbaa !4
  %2505 = fmul float %2503, %2504
  store float %2505, ptr %1281, align 4, !tbaa !4
  %2506 = getelementptr inbounds nuw i8, ptr %2215, i64 264
  %2507 = load float, ptr %2506, align 4, !tbaa !4
  %2508 = load float, ptr %1487, align 8, !tbaa !4
  %2509 = fmul float %2507, %2508
  store float %2509, ptr %1282, align 8, !tbaa !4
  %2510 = getelementptr inbounds nuw i8, ptr %2215, i64 268
  %2511 = load float, ptr %2510, align 4, !tbaa !4
  %2512 = load float, ptr %1488, align 4, !tbaa !4
  %2513 = fmul float %2511, %2512
  store float %2513, ptr %1283, align 4, !tbaa !4
  %2514 = getelementptr inbounds nuw i8, ptr %2215, i64 272
  %2515 = load float, ptr %2514, align 4, !tbaa !4
  %2516 = load float, ptr %1489, align 16, !tbaa !4
  %2517 = fmul float %2515, %2516
  store float %2517, ptr %1284, align 16, !tbaa !4
  %2518 = getelementptr inbounds nuw i8, ptr %2215, i64 276
  %2519 = load float, ptr %2518, align 4, !tbaa !4
  %2520 = load float, ptr %1490, align 4, !tbaa !4
  %2521 = fmul float %2519, %2520
  store float %2521, ptr %1285, align 4, !tbaa !4
  %2522 = getelementptr inbounds nuw i8, ptr %2215, i64 280
  %2523 = load float, ptr %2522, align 4, !tbaa !4
  %2524 = load float, ptr %1491, align 8, !tbaa !4
  %2525 = fmul float %2523, %2524
  store float %2525, ptr %1286, align 8, !tbaa !4
  %2526 = getelementptr inbounds nuw i8, ptr %2215, i64 284
  %2527 = load float, ptr %2526, align 4, !tbaa !4
  %2528 = load float, ptr %1492, align 4, !tbaa !4
  %2529 = fmul float %2527, %2528
  store float %2529, ptr %1287, align 4, !tbaa !4
  %2530 = getelementptr inbounds nuw i8, ptr %2215, i64 288
  %2531 = load float, ptr %2530, align 4, !tbaa !4
  %2532 = load float, ptr %1493, align 32, !tbaa !4
  %2533 = fmul float %2531, %2532
  store float %2533, ptr %1288, align 32, !tbaa !4
  %2534 = getelementptr inbounds nuw i8, ptr %2215, i64 292
  %2535 = load float, ptr %2534, align 4, !tbaa !4
  %2536 = load float, ptr %1494, align 4, !tbaa !4
  %2537 = fmul float %2535, %2536
  store float %2537, ptr %1289, align 4, !tbaa !4
  %2538 = getelementptr inbounds nuw i8, ptr %2215, i64 296
  %2539 = load float, ptr %2538, align 4, !tbaa !4
  %2540 = load float, ptr %1495, align 8, !tbaa !4
  %2541 = fmul float %2539, %2540
  store float %2541, ptr %1290, align 8, !tbaa !4
  %2542 = getelementptr inbounds nuw i8, ptr %2215, i64 300
  %2543 = load float, ptr %2542, align 4, !tbaa !4
  %2544 = load float, ptr %1496, align 4, !tbaa !4
  %2545 = fmul float %2543, %2544
  store float %2545, ptr %1291, align 4, !tbaa !4
  %2546 = getelementptr inbounds nuw i8, ptr %2215, i64 304
  %2547 = load float, ptr %2546, align 4, !tbaa !4
  %2548 = load float, ptr %1497, align 16, !tbaa !4
  %2549 = fmul float %2547, %2548
  store float %2549, ptr %1292, align 16, !tbaa !4
  %2550 = getelementptr inbounds nuw i8, ptr %2215, i64 308
  %2551 = load float, ptr %2550, align 4, !tbaa !4
  %2552 = load float, ptr %1498, align 4, !tbaa !4
  %2553 = fmul float %2551, %2552
  store float %2553, ptr %1293, align 4, !tbaa !4
  %2554 = getelementptr inbounds nuw i8, ptr %2215, i64 312
  %2555 = load float, ptr %2554, align 4, !tbaa !4
  %2556 = load float, ptr %1499, align 8, !tbaa !4
  %2557 = fmul float %2555, %2556
  store float %2557, ptr %1294, align 8, !tbaa !4
  %2558 = getelementptr inbounds nuw i8, ptr %2215, i64 316
  %2559 = load float, ptr %2558, align 4, !tbaa !4
  %2560 = load float, ptr %1500, align 4, !tbaa !4
  %2561 = fmul float %2559, %2560
  store float %2561, ptr %1295, align 4, !tbaa !4
  %2562 = getelementptr inbounds nuw i8, ptr %2215, i64 320
  %2563 = load float, ptr %2562, align 4, !tbaa !4
  %2564 = load float, ptr %1501, align 64, !tbaa !4
  %2565 = fmul float %2563, %2564
  store float %2565, ptr %1296, align 64, !tbaa !4
  %2566 = getelementptr inbounds nuw i8, ptr %2215, i64 324
  %2567 = load float, ptr %2566, align 4, !tbaa !4
  %2568 = load float, ptr %1502, align 4, !tbaa !4
  %2569 = fmul float %2567, %2568
  store float %2569, ptr %1297, align 4, !tbaa !4
  %2570 = getelementptr inbounds nuw i8, ptr %2215, i64 328
  %2571 = load float, ptr %2570, align 4, !tbaa !4
  %2572 = load float, ptr %1503, align 8, !tbaa !4
  %2573 = fmul float %2571, %2572
  store float %2573, ptr %1298, align 8, !tbaa !4
  %2574 = getelementptr inbounds nuw i8, ptr %2215, i64 332
  %2575 = load float, ptr %2574, align 4, !tbaa !4
  %2576 = load float, ptr %1504, align 4, !tbaa !4
  %2577 = fmul float %2575, %2576
  store float %2577, ptr %1299, align 4, !tbaa !4
  %2578 = getelementptr inbounds nuw i8, ptr %2215, i64 336
  %2579 = load float, ptr %2578, align 4, !tbaa !4
  %2580 = load float, ptr %1505, align 16, !tbaa !4
  %2581 = fmul float %2579, %2580
  store float %2581, ptr %1300, align 16, !tbaa !4
  %2582 = getelementptr inbounds nuw i8, ptr %2215, i64 340
  %2583 = load float, ptr %2582, align 4, !tbaa !4
  %2584 = load float, ptr %1506, align 4, !tbaa !4
  %2585 = fmul float %2583, %2584
  store float %2585, ptr %1301, align 4, !tbaa !4
  %2586 = getelementptr inbounds nuw i8, ptr %2215, i64 344
  %2587 = load float, ptr %2586, align 4, !tbaa !4
  %2588 = load float, ptr %1507, align 8, !tbaa !4
  %2589 = fmul float %2587, %2588
  store float %2589, ptr %1302, align 8, !tbaa !4
  %2590 = getelementptr inbounds nuw i8, ptr %2215, i64 348
  %2591 = load float, ptr %2590, align 4, !tbaa !4
  %2592 = load float, ptr %1508, align 4, !tbaa !4
  %2593 = fmul float %2591, %2592
  store float %2593, ptr %1303, align 4, !tbaa !4
  %2594 = getelementptr inbounds nuw i8, ptr %2215, i64 352
  %2595 = load float, ptr %2594, align 4, !tbaa !4
  %2596 = load float, ptr %1509, align 32, !tbaa !4
  %2597 = fmul float %2595, %2596
  store float %2597, ptr %1304, align 32, !tbaa !4
  %2598 = getelementptr inbounds nuw i8, ptr %2215, i64 356
  %2599 = load float, ptr %2598, align 4, !tbaa !4
  %2600 = load float, ptr %1510, align 4, !tbaa !4
  %2601 = fmul float %2599, %2600
  store float %2601, ptr %1305, align 4, !tbaa !4
  %2602 = getelementptr inbounds nuw i8, ptr %2215, i64 360
  %2603 = load float, ptr %2602, align 4, !tbaa !4
  %2604 = load float, ptr %1511, align 8, !tbaa !4
  %2605 = fmul float %2603, %2604
  store float %2605, ptr %1306, align 8, !tbaa !4
  %2606 = getelementptr inbounds nuw i8, ptr %2215, i64 364
  %2607 = load float, ptr %2606, align 4, !tbaa !4
  %2608 = load float, ptr %1512, align 4, !tbaa !4
  %2609 = fmul float %2607, %2608
  store float %2609, ptr %1307, align 4, !tbaa !4
  %2610 = getelementptr inbounds nuw i8, ptr %2215, i64 368
  %2611 = load float, ptr %2610, align 4, !tbaa !4
  %2612 = load float, ptr %1513, align 16, !tbaa !4
  %2613 = fmul float %2611, %2612
  store float %2613, ptr %1308, align 16, !tbaa !4
  %2614 = getelementptr inbounds nuw i8, ptr %2215, i64 372
  %2615 = load float, ptr %2614, align 4, !tbaa !4
  %2616 = load float, ptr %1514, align 4, !tbaa !4
  %2617 = fmul float %2615, %2616
  store float %2617, ptr %1309, align 4, !tbaa !4
  %2618 = getelementptr inbounds nuw i8, ptr %2215, i64 376
  %2619 = load float, ptr %2618, align 4, !tbaa !4
  %2620 = load float, ptr %1515, align 8, !tbaa !4
  %2621 = fmul float %2619, %2620
  store float %2621, ptr %1310, align 8, !tbaa !4
  %2622 = getelementptr inbounds nuw i8, ptr %2215, i64 380
  %2623 = load float, ptr %2622, align 4, !tbaa !4
  %2624 = load float, ptr %1516, align 4, !tbaa !4
  %2625 = fmul float %2623, %2624
  store float %2625, ptr %1311, align 4, !tbaa !4
  tail call void @_mlir_memref_to_llvm_free(ptr nonnull %2215)
  br label %.preheader336

.preheader336:                                    ; preds = %.preheader337.preheader, %.preheader336
  %2626 = phi i64 [ 0, %.preheader337.preheader ], [ %2651, %.preheader336 ]
  %.idx = mul nuw nsw i64 %2626, 96
  %2627 = getelementptr i8, ptr %1321, i64 %.idx
  store float 0x3F847AE140000000, ptr %2627, align 32, !tbaa !4
  %2628 = getelementptr i8, ptr %2627, i64 4
  store float 0x3F847AE140000000, ptr %2628, align 4, !tbaa !4
  %2629 = getelementptr i8, ptr %2627, i64 8
  store float 0x3F847AE140000000, ptr %2629, align 8, !tbaa !4
  %2630 = getelementptr i8, ptr %2627, i64 12
  store float 0x3F847AE140000000, ptr %2630, align 4, !tbaa !4
  %2631 = getelementptr i8, ptr %2627, i64 16
  store float 0x3F847AE140000000, ptr %2631, align 16, !tbaa !4
  %2632 = getelementptr i8, ptr %2627, i64 20
  store float 0x3F847AE140000000, ptr %2632, align 4, !tbaa !4
  %2633 = getelementptr i8, ptr %2627, i64 24
  store float 0x3F847AE140000000, ptr %2633, align 8, !tbaa !4
  %2634 = getelementptr i8, ptr %2627, i64 28
  store float 0x3F847AE140000000, ptr %2634, align 4, !tbaa !4
  %2635 = getelementptr i8, ptr %2627, i64 32
  store float 0x3F847AE140000000, ptr %2635, align 32, !tbaa !4
  %2636 = getelementptr i8, ptr %2627, i64 36
  store float 0x3F847AE140000000, ptr %2636, align 4, !tbaa !4
  %2637 = getelementptr i8, ptr %2627, i64 40
  store float 0x3F847AE140000000, ptr %2637, align 8, !tbaa !4
  %2638 = getelementptr i8, ptr %2627, i64 44
  store float 0x3F847AE140000000, ptr %2638, align 4, !tbaa !4
  %2639 = getelementptr i8, ptr %2627, i64 48
  store float 0x3F847AE140000000, ptr %2639, align 16, !tbaa !4
  %2640 = getelementptr i8, ptr %2627, i64 52
  store float 0x3F847AE140000000, ptr %2640, align 4, !tbaa !4
  %2641 = getelementptr i8, ptr %2627, i64 56
  store float 0x3F847AE140000000, ptr %2641, align 8, !tbaa !4
  %2642 = getelementptr i8, ptr %2627, i64 60
  store float 0x3F847AE140000000, ptr %2642, align 4, !tbaa !4
  %2643 = getelementptr i8, ptr %2627, i64 64
  store float 0x3F847AE140000000, ptr %2643, align 32, !tbaa !4
  %2644 = getelementptr i8, ptr %2627, i64 68
  store float 0x3F847AE140000000, ptr %2644, align 4, !tbaa !4
  %2645 = getelementptr i8, ptr %2627, i64 72
  store float 0x3F847AE140000000, ptr %2645, align 8, !tbaa !4
  %2646 = getelementptr i8, ptr %2627, i64 76
  store float 0x3F847AE140000000, ptr %2646, align 4, !tbaa !4
  %2647 = getelementptr i8, ptr %2627, i64 80
  store float 0x3F847AE140000000, ptr %2647, align 16, !tbaa !4
  %2648 = getelementptr i8, ptr %2627, i64 84
  store float 0x3F847AE140000000, ptr %2648, align 4, !tbaa !4
  %2649 = getelementptr i8, ptr %2627, i64 88
  store float 0x3F847AE140000000, ptr %2649, align 8, !tbaa !4
  %2650 = getelementptr i8, ptr %2627, i64 92
  store float 0x3F847AE140000000, ptr %2650, align 4, !tbaa !4
  %2651 = add nuw nsw i64 %2626, 1
  %exitcond472.not = icmp eq i64 %2651, 4
  br i1 %exitcond472.not, label %.preheader335, label %.preheader336

.preheader335:                                    ; preds = %.preheader336, %.preheader335
  %2652 = phi i64 [ %2797, %.preheader335 ], [ 0, %.preheader336 ]
  %2653 = mul nuw nsw i64 %2652, 24
  %2654 = getelementptr inbounds nuw float, ptr %1321, i64 %2653
  %2655 = load float, ptr %2654, align 32, !tbaa !4
  %2656 = getelementptr inbounds nuw float, ptr %1109, i64 %2653
  %2657 = load float, ptr %2656, align 32, !tbaa !4
  %2658 = fmul float %2655, %2657
  store float %2658, ptr %2654, align 32, !tbaa !4
  %2659 = or disjoint i64 %2653, 1
  %2660 = getelementptr inbounds nuw float, ptr %1321, i64 %2659
  %2661 = load float, ptr %2660, align 4, !tbaa !4
  %2662 = getelementptr inbounds nuw float, ptr %1109, i64 %2659
  %2663 = load float, ptr %2662, align 4, !tbaa !4
  %2664 = fmul float %2661, %2663
  store float %2664, ptr %2660, align 4, !tbaa !4
  %2665 = or disjoint i64 %2653, 2
  %2666 = getelementptr inbounds nuw float, ptr %1321, i64 %2665
  %2667 = load float, ptr %2666, align 8, !tbaa !4
  %2668 = getelementptr inbounds nuw float, ptr %1109, i64 %2665
  %2669 = load float, ptr %2668, align 8, !tbaa !4
  %2670 = fmul float %2667, %2669
  store float %2670, ptr %2666, align 8, !tbaa !4
  %2671 = or disjoint i64 %2653, 3
  %2672 = getelementptr inbounds nuw float, ptr %1321, i64 %2671
  %2673 = load float, ptr %2672, align 4, !tbaa !4
  %2674 = getelementptr inbounds nuw float, ptr %1109, i64 %2671
  %2675 = load float, ptr %2674, align 4, !tbaa !4
  %2676 = fmul float %2673, %2675
  store float %2676, ptr %2672, align 4, !tbaa !4
  %2677 = or disjoint i64 %2653, 4
  %2678 = getelementptr inbounds nuw float, ptr %1321, i64 %2677
  %2679 = load float, ptr %2678, align 16, !tbaa !4
  %2680 = getelementptr inbounds nuw float, ptr %1109, i64 %2677
  %2681 = load float, ptr %2680, align 16, !tbaa !4
  %2682 = fmul float %2679, %2681
  store float %2682, ptr %2678, align 16, !tbaa !4
  %2683 = or disjoint i64 %2653, 5
  %2684 = getelementptr inbounds nuw float, ptr %1321, i64 %2683
  %2685 = load float, ptr %2684, align 4, !tbaa !4
  %2686 = getelementptr inbounds nuw float, ptr %1109, i64 %2683
  %2687 = load float, ptr %2686, align 4, !tbaa !4
  %2688 = fmul float %2685, %2687
  store float %2688, ptr %2684, align 4, !tbaa !4
  %2689 = or disjoint i64 %2653, 6
  %2690 = getelementptr inbounds nuw float, ptr %1321, i64 %2689
  %2691 = load float, ptr %2690, align 8, !tbaa !4
  %2692 = getelementptr inbounds nuw float, ptr %1109, i64 %2689
  %2693 = load float, ptr %2692, align 8, !tbaa !4
  %2694 = fmul float %2691, %2693
  store float %2694, ptr %2690, align 8, !tbaa !4
  %2695 = or disjoint i64 %2653, 7
  %2696 = getelementptr inbounds nuw float, ptr %1321, i64 %2695
  %2697 = load float, ptr %2696, align 4, !tbaa !4
  %2698 = getelementptr inbounds nuw float, ptr %1109, i64 %2695
  %2699 = load float, ptr %2698, align 4, !tbaa !4
  %2700 = fmul float %2697, %2699
  store float %2700, ptr %2696, align 4, !tbaa !4
  %2701 = add nuw nsw i64 %2653, 8
  %2702 = getelementptr inbounds nuw float, ptr %1321, i64 %2701
  %2703 = load float, ptr %2702, align 32, !tbaa !4
  %2704 = getelementptr inbounds nuw float, ptr %1109, i64 %2701
  %2705 = load float, ptr %2704, align 32, !tbaa !4
  %2706 = fmul float %2703, %2705
  store float %2706, ptr %2702, align 32, !tbaa !4
  %2707 = add nuw nsw i64 %2653, 9
  %2708 = getelementptr inbounds nuw float, ptr %1321, i64 %2707
  %2709 = load float, ptr %2708, align 4, !tbaa !4
  %2710 = getelementptr inbounds nuw float, ptr %1109, i64 %2707
  %2711 = load float, ptr %2710, align 4, !tbaa !4
  %2712 = fmul float %2709, %2711
  store float %2712, ptr %2708, align 4, !tbaa !4
  %2713 = add nuw nsw i64 %2653, 10
  %2714 = getelementptr inbounds nuw float, ptr %1321, i64 %2713
  %2715 = load float, ptr %2714, align 8, !tbaa !4
  %2716 = getelementptr inbounds nuw float, ptr %1109, i64 %2713
  %2717 = load float, ptr %2716, align 8, !tbaa !4
  %2718 = fmul float %2715, %2717
  store float %2718, ptr %2714, align 8, !tbaa !4
  %2719 = add nuw nsw i64 %2653, 11
  %2720 = getelementptr inbounds nuw float, ptr %1321, i64 %2719
  %2721 = load float, ptr %2720, align 4, !tbaa !4
  %2722 = getelementptr inbounds nuw float, ptr %1109, i64 %2719
  %2723 = load float, ptr %2722, align 4, !tbaa !4
  %2724 = fmul float %2721, %2723
  store float %2724, ptr %2720, align 4, !tbaa !4
  %2725 = add nuw nsw i64 %2653, 12
  %2726 = getelementptr inbounds nuw float, ptr %1321, i64 %2725
  %2727 = load float, ptr %2726, align 16, !tbaa !4
  %2728 = getelementptr inbounds nuw float, ptr %1109, i64 %2725
  %2729 = load float, ptr %2728, align 16, !tbaa !4
  %2730 = fmul float %2727, %2729
  store float %2730, ptr %2726, align 16, !tbaa !4
  %2731 = add nuw nsw i64 %2653, 13
  %2732 = getelementptr inbounds nuw float, ptr %1321, i64 %2731
  %2733 = load float, ptr %2732, align 4, !tbaa !4
  %2734 = getelementptr inbounds nuw float, ptr %1109, i64 %2731
  %2735 = load float, ptr %2734, align 4, !tbaa !4
  %2736 = fmul float %2733, %2735
  store float %2736, ptr %2732, align 4, !tbaa !4
  %2737 = add nuw nsw i64 %2653, 14
  %2738 = getelementptr inbounds nuw float, ptr %1321, i64 %2737
  %2739 = load float, ptr %2738, align 8, !tbaa !4
  %2740 = getelementptr inbounds nuw float, ptr %1109, i64 %2737
  %2741 = load float, ptr %2740, align 8, !tbaa !4
  %2742 = fmul float %2739, %2741
  store float %2742, ptr %2738, align 8, !tbaa !4
  %2743 = add nuw nsw i64 %2653, 15
  %2744 = getelementptr inbounds nuw float, ptr %1321, i64 %2743
  %2745 = load float, ptr %2744, align 4, !tbaa !4
  %2746 = getelementptr inbounds nuw float, ptr %1109, i64 %2743
  %2747 = load float, ptr %2746, align 4, !tbaa !4
  %2748 = fmul float %2745, %2747
  store float %2748, ptr %2744, align 4, !tbaa !4
  %2749 = add nuw nsw i64 %2653, 16
  %2750 = getelementptr inbounds nuw float, ptr %1321, i64 %2749
  %2751 = load float, ptr %2750, align 32, !tbaa !4
  %2752 = getelementptr inbounds nuw float, ptr %1109, i64 %2749
  %2753 = load float, ptr %2752, align 32, !tbaa !4
  %2754 = fmul float %2751, %2753
  store float %2754, ptr %2750, align 32, !tbaa !4
  %2755 = add nuw nsw i64 %2653, 17
  %2756 = getelementptr inbounds nuw float, ptr %1321, i64 %2755
  %2757 = load float, ptr %2756, align 4, !tbaa !4
  %2758 = getelementptr inbounds nuw float, ptr %1109, i64 %2755
  %2759 = load float, ptr %2758, align 4, !tbaa !4
  %2760 = fmul float %2757, %2759
  store float %2760, ptr %2756, align 4, !tbaa !4
  %2761 = add nuw nsw i64 %2653, 18
  %2762 = getelementptr inbounds nuw float, ptr %1321, i64 %2761
  %2763 = load float, ptr %2762, align 8, !tbaa !4
  %2764 = getelementptr inbounds nuw float, ptr %1109, i64 %2761
  %2765 = load float, ptr %2764, align 8, !tbaa !4
  %2766 = fmul float %2763, %2765
  store float %2766, ptr %2762, align 8, !tbaa !4
  %2767 = add nuw nsw i64 %2653, 19
  %2768 = getelementptr inbounds nuw float, ptr %1321, i64 %2767
  %2769 = load float, ptr %2768, align 4, !tbaa !4
  %2770 = getelementptr inbounds nuw float, ptr %1109, i64 %2767
  %2771 = load float, ptr %2770, align 4, !tbaa !4
  %2772 = fmul float %2769, %2771
  store float %2772, ptr %2768, align 4, !tbaa !4
  %2773 = add nuw nsw i64 %2653, 20
  %2774 = getelementptr inbounds nuw float, ptr %1321, i64 %2773
  %2775 = load float, ptr %2774, align 16, !tbaa !4
  %2776 = getelementptr inbounds nuw float, ptr %1109, i64 %2773
  %2777 = load float, ptr %2776, align 16, !tbaa !4
  %2778 = fmul float %2775, %2777
  store float %2778, ptr %2774, align 16, !tbaa !4
  %2779 = add nuw nsw i64 %2653, 21
  %2780 = getelementptr inbounds nuw float, ptr %1321, i64 %2779
  %2781 = load float, ptr %2780, align 4, !tbaa !4
  %2782 = getelementptr inbounds nuw float, ptr %1109, i64 %2779
  %2783 = load float, ptr %2782, align 4, !tbaa !4
  %2784 = fmul float %2781, %2783
  store float %2784, ptr %2780, align 4, !tbaa !4
  %2785 = add nuw nsw i64 %2653, 22
  %2786 = getelementptr inbounds nuw float, ptr %1321, i64 %2785
  %2787 = load float, ptr %2786, align 8, !tbaa !4
  %2788 = getelementptr inbounds nuw float, ptr %1109, i64 %2785
  %2789 = load float, ptr %2788, align 8, !tbaa !4
  %2790 = fmul float %2787, %2789
  store float %2790, ptr %2786, align 8, !tbaa !4
  %2791 = add nuw nsw i64 %2653, 23
  %2792 = getelementptr inbounds nuw float, ptr %1321, i64 %2791
  %2793 = load float, ptr %2792, align 4, !tbaa !4
  %2794 = getelementptr inbounds nuw float, ptr %1109, i64 %2791
  %2795 = load float, ptr %2794, align 4, !tbaa !4
  %2796 = fmul float %2793, %2795
  store float %2796, ptr %2792, align 4, !tbaa !4
  %2797 = add nuw nsw i64 %2652, 1
  %exitcond475.not = icmp eq i64 %2797, 4
  br i1 %exitcond475.not, label %.preheader334, label %.preheader335

.preheader334:                                    ; preds = %.preheader335, %.preheader334
  %2798 = phi i64 [ %2943, %.preheader334 ], [ 0, %.preheader335 ]
  %2799 = mul nuw nsw i64 %2798, 24
  %2800 = getelementptr inbounds nuw float, ptr %.pn241419, i64 %2799
  %2801 = load float, ptr %2800, align 4, !tbaa !4
  %2802 = getelementptr inbounds nuw float, ptr %1321, i64 %2799
  %2803 = load float, ptr %2802, align 32, !tbaa !4
  %2804 = fsub float %2801, %2803
  store float %2804, ptr %2802, align 32, !tbaa !4
  %2805 = or disjoint i64 %2799, 1
  %2806 = getelementptr inbounds nuw float, ptr %.pn241419, i64 %2805
  %2807 = load float, ptr %2806, align 4, !tbaa !4
  %2808 = getelementptr inbounds nuw float, ptr %1321, i64 %2805
  %2809 = load float, ptr %2808, align 4, !tbaa !4
  %2810 = fsub float %2807, %2809
  store float %2810, ptr %2808, align 4, !tbaa !4
  %2811 = or disjoint i64 %2799, 2
  %2812 = getelementptr inbounds nuw float, ptr %.pn241419, i64 %2811
  %2813 = load float, ptr %2812, align 4, !tbaa !4
  %2814 = getelementptr inbounds nuw float, ptr %1321, i64 %2811
  %2815 = load float, ptr %2814, align 8, !tbaa !4
  %2816 = fsub float %2813, %2815
  store float %2816, ptr %2814, align 8, !tbaa !4
  %2817 = or disjoint i64 %2799, 3
  %2818 = getelementptr inbounds nuw float, ptr %.pn241419, i64 %2817
  %2819 = load float, ptr %2818, align 4, !tbaa !4
  %2820 = getelementptr inbounds nuw float, ptr %1321, i64 %2817
  %2821 = load float, ptr %2820, align 4, !tbaa !4
  %2822 = fsub float %2819, %2821
  store float %2822, ptr %2820, align 4, !tbaa !4
  %2823 = or disjoint i64 %2799, 4
  %2824 = getelementptr inbounds nuw float, ptr %.pn241419, i64 %2823
  %2825 = load float, ptr %2824, align 4, !tbaa !4
  %2826 = getelementptr inbounds nuw float, ptr %1321, i64 %2823
  %2827 = load float, ptr %2826, align 16, !tbaa !4
  %2828 = fsub float %2825, %2827
  store float %2828, ptr %2826, align 16, !tbaa !4
  %2829 = or disjoint i64 %2799, 5
  %2830 = getelementptr inbounds nuw float, ptr %.pn241419, i64 %2829
  %2831 = load float, ptr %2830, align 4, !tbaa !4
  %2832 = getelementptr inbounds nuw float, ptr %1321, i64 %2829
  %2833 = load float, ptr %2832, align 4, !tbaa !4
  %2834 = fsub float %2831, %2833
  store float %2834, ptr %2832, align 4, !tbaa !4
  %2835 = or disjoint i64 %2799, 6
  %2836 = getelementptr inbounds nuw float, ptr %.pn241419, i64 %2835
  %2837 = load float, ptr %2836, align 4, !tbaa !4
  %2838 = getelementptr inbounds nuw float, ptr %1321, i64 %2835
  %2839 = load float, ptr %2838, align 8, !tbaa !4
  %2840 = fsub float %2837, %2839
  store float %2840, ptr %2838, align 8, !tbaa !4
  %2841 = or disjoint i64 %2799, 7
  %2842 = getelementptr inbounds nuw float, ptr %.pn241419, i64 %2841
  %2843 = load float, ptr %2842, align 4, !tbaa !4
  %2844 = getelementptr inbounds nuw float, ptr %1321, i64 %2841
  %2845 = load float, ptr %2844, align 4, !tbaa !4
  %2846 = fsub float %2843, %2845
  store float %2846, ptr %2844, align 4, !tbaa !4
  %2847 = add nuw nsw i64 %2799, 8
  %2848 = getelementptr inbounds nuw float, ptr %.pn241419, i64 %2847
  %2849 = load float, ptr %2848, align 4, !tbaa !4
  %2850 = getelementptr inbounds nuw float, ptr %1321, i64 %2847
  %2851 = load float, ptr %2850, align 32, !tbaa !4
  %2852 = fsub float %2849, %2851
  store float %2852, ptr %2850, align 32, !tbaa !4
  %2853 = add nuw nsw i64 %2799, 9
  %2854 = getelementptr inbounds nuw float, ptr %.pn241419, i64 %2853
  %2855 = load float, ptr %2854, align 4, !tbaa !4
  %2856 = getelementptr inbounds nuw float, ptr %1321, i64 %2853
  %2857 = load float, ptr %2856, align 4, !tbaa !4
  %2858 = fsub float %2855, %2857
  store float %2858, ptr %2856, align 4, !tbaa !4
  %2859 = add nuw nsw i64 %2799, 10
  %2860 = getelementptr inbounds nuw float, ptr %.pn241419, i64 %2859
  %2861 = load float, ptr %2860, align 4, !tbaa !4
  %2862 = getelementptr inbounds nuw float, ptr %1321, i64 %2859
  %2863 = load float, ptr %2862, align 8, !tbaa !4
  %2864 = fsub float %2861, %2863
  store float %2864, ptr %2862, align 8, !tbaa !4
  %2865 = add nuw nsw i64 %2799, 11
  %2866 = getelementptr inbounds nuw float, ptr %.pn241419, i64 %2865
  %2867 = load float, ptr %2866, align 4, !tbaa !4
  %2868 = getelementptr inbounds nuw float, ptr %1321, i64 %2865
  %2869 = load float, ptr %2868, align 4, !tbaa !4
  %2870 = fsub float %2867, %2869
  store float %2870, ptr %2868, align 4, !tbaa !4
  %2871 = add nuw nsw i64 %2799, 12
  %2872 = getelementptr inbounds nuw float, ptr %.pn241419, i64 %2871
  %2873 = load float, ptr %2872, align 4, !tbaa !4
  %2874 = getelementptr inbounds nuw float, ptr %1321, i64 %2871
  %2875 = load float, ptr %2874, align 16, !tbaa !4
  %2876 = fsub float %2873, %2875
  store float %2876, ptr %2874, align 16, !tbaa !4
  %2877 = add nuw nsw i64 %2799, 13
  %2878 = getelementptr inbounds nuw float, ptr %.pn241419, i64 %2877
  %2879 = load float, ptr %2878, align 4, !tbaa !4
  %2880 = getelementptr inbounds nuw float, ptr %1321, i64 %2877
  %2881 = load float, ptr %2880, align 4, !tbaa !4
  %2882 = fsub float %2879, %2881
  store float %2882, ptr %2880, align 4, !tbaa !4
  %2883 = add nuw nsw i64 %2799, 14
  %2884 = getelementptr inbounds nuw float, ptr %.pn241419, i64 %2883
  %2885 = load float, ptr %2884, align 4, !tbaa !4
  %2886 = getelementptr inbounds nuw float, ptr %1321, i64 %2883
  %2887 = load float, ptr %2886, align 8, !tbaa !4
  %2888 = fsub float %2885, %2887
  store float %2888, ptr %2886, align 8, !tbaa !4
  %2889 = add nuw nsw i64 %2799, 15
  %2890 = getelementptr inbounds nuw float, ptr %.pn241419, i64 %2889
  %2891 = load float, ptr %2890, align 4, !tbaa !4
  %2892 = getelementptr inbounds nuw float, ptr %1321, i64 %2889
  %2893 = load float, ptr %2892, align 4, !tbaa !4
  %2894 = fsub float %2891, %2893
  store float %2894, ptr %2892, align 4, !tbaa !4
  %2895 = add nuw nsw i64 %2799, 16
  %2896 = getelementptr inbounds nuw float, ptr %.pn241419, i64 %2895
  %2897 = load float, ptr %2896, align 4, !tbaa !4
  %2898 = getelementptr inbounds nuw float, ptr %1321, i64 %2895
  %2899 = load float, ptr %2898, align 32, !tbaa !4
  %2900 = fsub float %2897, %2899
  store float %2900, ptr %2898, align 32, !tbaa !4
  %2901 = add nuw nsw i64 %2799, 17
  %2902 = getelementptr inbounds nuw float, ptr %.pn241419, i64 %2901
  %2903 = load float, ptr %2902, align 4, !tbaa !4
  %2904 = getelementptr inbounds nuw float, ptr %1321, i64 %2901
  %2905 = load float, ptr %2904, align 4, !tbaa !4
  %2906 = fsub float %2903, %2905
  store float %2906, ptr %2904, align 4, !tbaa !4
  %2907 = add nuw nsw i64 %2799, 18
  %2908 = getelementptr inbounds nuw float, ptr %.pn241419, i64 %2907
  %2909 = load float, ptr %2908, align 4, !tbaa !4
  %2910 = getelementptr inbounds nuw float, ptr %1321, i64 %2907
  %2911 = load float, ptr %2910, align 8, !tbaa !4
  %2912 = fsub float %2909, %2911
  store float %2912, ptr %2910, align 8, !tbaa !4
  %2913 = add nuw nsw i64 %2799, 19
  %2914 = getelementptr inbounds nuw float, ptr %.pn241419, i64 %2913
  %2915 = load float, ptr %2914, align 4, !tbaa !4
  %2916 = getelementptr inbounds nuw float, ptr %1321, i64 %2913
  %2917 = load float, ptr %2916, align 4, !tbaa !4
  %2918 = fsub float %2915, %2917
  store float %2918, ptr %2916, align 4, !tbaa !4
  %2919 = add nuw nsw i64 %2799, 20
  %2920 = getelementptr inbounds nuw float, ptr %.pn241419, i64 %2919
  %2921 = load float, ptr %2920, align 4, !tbaa !4
  %2922 = getelementptr inbounds nuw float, ptr %1321, i64 %2919
  %2923 = load float, ptr %2922, align 16, !tbaa !4
  %2924 = fsub float %2921, %2923
  store float %2924, ptr %2922, align 16, !tbaa !4
  %2925 = add nuw nsw i64 %2799, 21
  %2926 = getelementptr inbounds nuw float, ptr %.pn241419, i64 %2925
  %2927 = load float, ptr %2926, align 4, !tbaa !4
  %2928 = getelementptr inbounds nuw float, ptr %1321, i64 %2925
  %2929 = load float, ptr %2928, align 4, !tbaa !4
  %2930 = fsub float %2927, %2929
  store float %2930, ptr %2928, align 4, !tbaa !4
  %2931 = add nuw nsw i64 %2799, 22
  %2932 = getelementptr inbounds nuw float, ptr %.pn241419, i64 %2931
  %2933 = load float, ptr %2932, align 4, !tbaa !4
  %2934 = getelementptr inbounds nuw float, ptr %1321, i64 %2931
  %2935 = load float, ptr %2934, align 8, !tbaa !4
  %2936 = fsub float %2933, %2935
  store float %2936, ptr %2934, align 8, !tbaa !4
  %2937 = add nuw nsw i64 %2799, 23
  %2938 = getelementptr inbounds nuw float, ptr %.pn241419, i64 %2937
  %2939 = load float, ptr %2938, align 4, !tbaa !4
  %2940 = getelementptr inbounds nuw float, ptr %1321, i64 %2937
  %2941 = load float, ptr %2940, align 4, !tbaa !4
  %2942 = fsub float %2939, %2941
  store float %2942, ptr %2940, align 4, !tbaa !4
  %2943 = add nuw nsw i64 %2798, 1
  %exitcond478.not = icmp eq i64 %2943, 4
  br i1 %exitcond478.not, label %2944, label %.preheader334

2944:                                             ; preds = %.preheader334
  tail call void @_mlir_memref_to_llvm_free(ptr nonnull %.pn241419)
  %2945 = fmul double %2206, 3.125000e-02
  %2946 = fptrunc double %2945 to float
  %2947 = fmul float %2946, 0x3F847AE140000000
  %2948 = fsub float %1315, %2947
  %2949 = fmul double %2208, 3.125000e-02
  %2950 = fptrunc double %2949 to float
  %2951 = fmul float %2950, 0x3F847AE140000000
  %2952 = fsub float %1314, %2951
  %2953 = fmul double %2209, 3.125000e-02
  %2954 = fadd double %1313, %2953
  %2955 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 384)
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(384) %2955, ptr noundef nonnull align 64 dereferenceable(384) %1321, i64 384, i1 false)
  tail call void @_mlir_memref_to_llvm_free(ptr %1317)
  %2956 = add nuw nsw i64 %1316, 1
  %exitcond479.not = icmp eq i64 %2956, 3
  br i1 %exitcond479.not, label %2957, label %1312

2957:                                             ; preds = %2944
  tail call void @_mlir_memref_to_llvm_free(ptr %1105)
  tail call void @_mlir_memref_to_llvm_free(ptr %1104)
  tail call void @_mlir_memref_to_llvm_free(ptr nonnull %1103)
  tail call void @_mlir_memref_to_llvm_free(ptr nonnull %1102)
  tail call void @_mlir_memref_to_llvm_free(ptr nonnull %1101)
  tail call void @_mlir_memref_to_llvm_free(ptr %1096)
  tail call void @_mlir_memref_to_llvm_free(ptr %1091)
  tail call void @_mlir_memref_to_llvm_free(ptr %1086)
  tail call void @_mlir_memref_to_llvm_free(ptr %1081)
  tail call void @_mlir_memref_to_llvm_free(ptr %1076)
  %2958 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 68)
  %2959 = ptrtoint ptr %2958 to i64
  %2960 = add i64 %2959, 63
  %2961 = and i64 %2960, -64
  %2962 = inttoptr i64 %2961 to ptr
  store float %2948, ptr %2962, align 64, !tbaa !4
  %2963 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 68)
  %2964 = ptrtoint ptr %2963 to i64
  %2965 = add i64 %2964, 63
  %2966 = and i64 %2965, -64
  %2967 = inttoptr i64 %2966 to ptr
  store float %2952, ptr %2967, align 64, !tbaa !4
  %2968 = fdiv double %2954, 3.000000e+00
  %2969 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %2970 = ptrtoint ptr %2969 to i64
  %2971 = add i64 %2970, 63
  %2972 = and i64 %2971, -64
  %2973 = inttoptr i64 %2972 to ptr
  store double %2968, ptr %2973, align 64, !tbaa !6
  %2974 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %2975 = ptrtoint ptr %2974 to i64
  %2976 = add i64 %2975, 63
  %2977 = and i64 %2976, -64
  %2978 = inttoptr i64 %2977 to ptr
  store double %2953, ptr %2978, align 64, !tbaa !6
  %2979 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 80)
  %2980 = ptrtoint ptr %2979 to i64
  %2981 = add i64 %2980, 63
  %2982 = and i64 %2981, -64
  %2983 = inttoptr i64 %2982 to ptr
  %2984 = load double, ptr %2973, align 64, !tbaa !6
  store double %2984, ptr %2983, align 64, !tbaa !6
  %2985 = load double, ptr %2978, align 64, !tbaa !6
  %2986 = getelementptr inbounds nuw i8, ptr %2983, i64 8
  store double %2985, ptr %2986, align 8, !tbaa !6
  tail call void @_mlir_memref_to_llvm_free(ptr %2974)
  tail call void @_mlir_memref_to_llvm_free(ptr %2969)
  %2987 = icmp eq ptr %2955, inttoptr (i64 3735928559 to ptr)
  br i1 %2987, label %2988, label %2990

2988:                                             ; preds = %2957
  %2989 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 384)
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(384) %2989, ptr noundef nonnull align 1 dereferenceable(384) inttoptr (i64 3735928559 to ptr), i64 384, i1 false)
  br label %2990

2990:                                             ; preds = %2988, %2957
  %.pn318 = phi ptr [ %2989, %2988 ], [ %2955, %2957 ]
  %2991 = icmp eq ptr %2958, inttoptr (i64 3735928559 to ptr)
  br i1 %2991, label %2992, label %2995

2992:                                             ; preds = %2990
  %2993 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 4)
  %2994 = load i32, ptr %2962, align 64
  store i32 %2994, ptr %2993, align 1
  br label %2995

2995:                                             ; preds = %2992, %2990
  %.pn249 = phi ptr [ %2993, %2992 ], [ %2958, %2990 ]
  %.pn247 = phi ptr [ %2993, %2992 ], [ %2962, %2990 ]
  %2996 = icmp eq ptr %2963, inttoptr (i64 3735928559 to ptr)
  br i1 %2996, label %2997, label %3000

2997:                                             ; preds = %2995
  %2998 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 4)
  %2999 = load i32, ptr %2967, align 64
  store i32 %2999, ptr %2998, align 1
  br label %3000

3000:                                             ; preds = %2997, %2995
  %.pn255 = phi ptr [ %2998, %2997 ], [ %2963, %2995 ]
  %.pn253 = phi ptr [ %2998, %2997 ], [ %2967, %2995 ]
  %3001 = icmp eq ptr %15, inttoptr (i64 3735928559 to ptr)
  br i1 %3001, label %3002, label %3006

3002:                                             ; preds = %3000
  %3003 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 4)
  %3004 = getelementptr inbounds i32, ptr %16, i64 %17
  %3005 = load i32, ptr %3004, align 1
  store i32 %3005, ptr %3003, align 1
  br label %3006

3006:                                             ; preds = %3002, %3000
  %.pn261 = phi ptr [ %3003, %3002 ], [ %15, %3000 ]
  %.pn259 = phi ptr [ %3003, %3002 ], [ %16, %3000 ]
  %.pn257 = phi i64 [ 0, %3002 ], [ %17, %3000 ]
  %3007 = icmp eq ptr %449, inttoptr (i64 3735928559 to ptr)
  br i1 %3007, label %3008, label %3011

3008:                                             ; preds = %3006
  %3009 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 8)
  %3010 = load i64, ptr %453, align 64
  store i64 %3010, ptr %3009, align 1
  br label %3011

3011:                                             ; preds = %3008, %3006
  %.pn271 = phi ptr [ %3009, %3008 ], [ %449, %3006 ]
  %.pn269 = phi ptr [ %3009, %3008 ], [ %453, %3006 ]
  %3012 = icmp eq ptr %2979, inttoptr (i64 3735928559 to ptr)
  br i1 %3012, label %3013, label %3015

3013:                                             ; preds = %3011
  %3014 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 16)
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(16) %3014, ptr noundef nonnull align 64 dereferenceable(16) %2983, i64 16, i1 false)
  br label %3015

3015:                                             ; preds = %3013, %3011
  %.pn281 = phi ptr [ %3014, %3013 ], [ %2979, %3011 ]
  %.pn279 = phi ptr [ %3014, %3013 ], [ %2983, %3011 ]
  %.pn317 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } poison, ptr %.pn318, 0
  %.pn315 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %.pn317, ptr %.pn318, 1
  %.pn313 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %.pn315, i64 0, 2
  %.pn311 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %.pn313, i64 4, 3, 0
  %.pn309 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %.pn311, i64 8, 3, 1
  %.pn307 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %.pn309, i64 3, 3, 2
  %.pn305 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %.pn307, i64 24, 4, 0
  %.pn303 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %.pn305, i64 3, 4, 1
  %3016 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %.pn303, i64 1, 4, 2
  %.pn278 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %.pn281, 0
  %.pn276 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %.pn278, ptr %.pn279, 1
  %.pn274 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %.pn276, i64 0, 2
  %.pn272 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %.pn274, i64 2, 3, 0
  %3017 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %.pn272, i64 1, 4, 0
  %.pn268 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %.pn271, 0
  %.pn266 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %.pn268, ptr %.pn269, 1
  %.pn264 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %.pn266, i64 0, 2
  %.pn262 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %.pn264, i64 2, 3, 0
  %3018 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %.pn262, i64 1, 4, 0
  %.pn258 = insertvalue { ptr, ptr, i64 } poison, ptr %.pn261, 0
  %.pn256 = insertvalue { ptr, ptr, i64 } %.pn258, ptr %.pn259, 1
  %3019 = insertvalue { ptr, ptr, i64 } %.pn256, i64 %.pn257, 2
  %.pn252 = insertvalue { ptr, ptr, i64 } poison, ptr %.pn255, 0
  %.pn250 = insertvalue { ptr, ptr, i64 } %.pn252, ptr %.pn253, 1
  %3020 = insertvalue { ptr, ptr, i64 } %.pn250, i64 0, 2
  %.pn246 = insertvalue { ptr, ptr, i64 } poison, ptr %.pn249, 0
  %.pn244 = insertvalue { ptr, ptr, i64 } %.pn246, ptr %.pn247, 1
  %3021 = insertvalue { ptr, ptr, i64 } %.pn244, i64 0, 2
  %3022 = insertvalue { { ptr, ptr, i64, [3 x i64], [3 x i64] }, { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64, [1 x i64], [1 x i64] }, { ptr, ptr, i64, [1 x i64], [1 x i64] } } poison, { ptr, ptr, i64, [3 x i64], [3 x i64] } %3016, 0
  %3023 = insertvalue { { ptr, ptr, i64, [3 x i64], [3 x i64] }, { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64, [1 x i64], [1 x i64] }, { ptr, ptr, i64, [1 x i64], [1 x i64] } } %3022, { ptr, ptr, i64 } %3021, 1
  %3024 = insertvalue { { ptr, ptr, i64, [3 x i64], [3 x i64] }, { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64, [1 x i64], [1 x i64] }, { ptr, ptr, i64, [1 x i64], [1 x i64] } } %3023, { ptr, ptr, i64 } %3020, 2
  %3025 = insertvalue { { ptr, ptr, i64, [3 x i64], [3 x i64] }, { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64, [1 x i64], [1 x i64] }, { ptr, ptr, i64, [1 x i64], [1 x i64] } } %3024, { ptr, ptr, i64 } %3019, 3
  %3026 = insertvalue { { ptr, ptr, i64, [3 x i64], [3 x i64] }, { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64, [1 x i64], [1 x i64] }, { ptr, ptr, i64, [1 x i64], [1 x i64] } } %3025, { ptr, ptr, i64, [1 x i64], [1 x i64] } %3018, 4
  %3027 = insertvalue { { ptr, ptr, i64, [3 x i64], [3 x i64] }, { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64, [1 x i64], [1 x i64] }, { ptr, ptr, i64, [1 x i64], [1 x i64] } } %3026, { ptr, ptr, i64, [1 x i64], [1 x i64] } %3017, 5
  ret { { ptr, ptr, i64, [3 x i64], [3 x i64] }, { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64, [1 x i64], [1 x i64] }, { ptr, ptr, i64, [1 x i64], [1 x i64] } } %3027
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
  %.unpack60 = load ptr, ptr %7, align 8
  %.elt61 = getelementptr inbounds nuw i8, ptr %7, i64 8
  %.unpack62 = load ptr, ptr %.elt61, align 8
  %.unpack71 = load ptr, ptr %8, align 8
  %.elt72 = getelementptr inbounds nuw i8, ptr %8, i64 8
  %.unpack73 = load ptr, ptr %.elt72, align 8
  %10 = tail call { { ptr, ptr, i64, [3 x i64], [3 x i64] }, { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64, [1 x i64], [1 x i64] }, { ptr, ptr, i64, [1 x i64], [1 x i64] } } @jit_train_epoch_compiled(ptr poison, ptr %.unpack2, i64 %.unpack4, i64 %.unpack6.unpack, i64 %.unpack6.unpack10, i64 %.unpack6.unpack12, i64 poison, i64 poison, i64 poison, ptr poison, ptr %.unpack21, i64 poison, ptr poison, ptr %.unpack26, i64 poison, ptr %.unpack29, ptr %.unpack31, i64 %.unpack33, ptr poison, ptr %.unpack36, i64 poison, i64 poison, i64 poison, ptr %.unpack45, ptr %.unpack47, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, ptr %.unpack60, ptr %.unpack62, i64 poison, i64 poison, i64 poison, ptr %.unpack71, ptr %.unpack73, i64 poison, i64 poison, i64 poison)
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
  %16 = load double, ptr %.unpack45, align 8, !tbaa !6
  %17 = getelementptr inbounds nuw double, ptr %11, i64 %15
  %18 = load double, ptr %17, align 8, !tbaa !6
  %19 = getelementptr inbounds nuw double, ptr %.unpack34, i64 %15
  %20 = load double, ptr %19, align 8, !tbaa !6
  %21 = fmul double %16, %18
  %22 = fadd double %20, %21
  store double %22, ptr %19, align 8, !tbaa !6
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
  tail call void @__catalyst__rt__device_init(ptr nonnull @"/home/pichau/QAgents/.venv/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.so", ptr nonnull @LightningSimulator, ptr nonnull @"{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}", i64 0, i1 false)
  %9 = tail call ptr @__catalyst__rt__qubit_allocate_array(i64 8)
  %10 = tail call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %9, i64 7)
  %11 = load ptr, ptr %10, align 8
  %12 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %13 = load double, ptr %12, align 8, !tbaa !6
  tail call void @__catalyst__qis__RY(double %13, ptr %11, ptr null)
  %14 = getelementptr inbounds nuw i8, ptr %12, i64 8
  %15 = load double, ptr %14, align 8, !tbaa !6
  tail call void @__catalyst__qis__RZ(double %15, ptr %11, ptr null)
  %16 = getelementptr inbounds nuw i8, ptr %12, i64 16
  %17 = load double, ptr %16, align 8, !tbaa !6
  tail call void @__catalyst__qis__RY(double %17, ptr %11, ptr null)
  %18 = getelementptr inbounds nuw i8, ptr %12, i64 24
  %19 = load double, ptr %18, align 8, !tbaa !6
  tail call void @__catalyst__qis__RZ(double %19, ptr %11, ptr null)
  %20 = tail call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %9, i64 6)
  %21 = load ptr, ptr %20, align 8
  %22 = getelementptr inbounds nuw i8, ptr %12, i64 32
  %23 = load double, ptr %22, align 8, !tbaa !6
  tail call void @__catalyst__qis__RY(double %23, ptr %21, ptr null)
  %24 = getelementptr inbounds nuw i8, ptr %12, i64 40
  %25 = load double, ptr %24, align 8, !tbaa !6
  tail call void @__catalyst__qis__RZ(double %25, ptr %21, ptr null)
  %26 = getelementptr inbounds nuw i8, ptr %12, i64 48
  %27 = load double, ptr %26, align 8, !tbaa !6
  tail call void @__catalyst__qis__RY(double %27, ptr %21, ptr null)
  %28 = getelementptr inbounds nuw i8, ptr %12, i64 56
  %29 = load double, ptr %28, align 8, !tbaa !6
  tail call void @__catalyst__qis__RZ(double %29, ptr %21, ptr null)
  %30 = tail call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %9, i64 5)
  %31 = load ptr, ptr %30, align 8
  %32 = getelementptr inbounds nuw i8, ptr %12, i64 64
  %33 = load double, ptr %32, align 8, !tbaa !6
  tail call void @__catalyst__qis__RY(double %33, ptr %31, ptr null)
  %34 = getelementptr inbounds nuw i8, ptr %12, i64 72
  %35 = load double, ptr %34, align 8, !tbaa !6
  tail call void @__catalyst__qis__RZ(double %35, ptr %31, ptr null)
  %36 = getelementptr inbounds nuw i8, ptr %12, i64 80
  %37 = load double, ptr %36, align 8, !tbaa !6
  tail call void @__catalyst__qis__RY(double %37, ptr %31, ptr null)
  %38 = getelementptr inbounds nuw i8, ptr %12, i64 88
  %39 = load double, ptr %38, align 8, !tbaa !6
  tail call void @__catalyst__qis__RZ(double %39, ptr %31, ptr null)
  %40 = tail call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %9, i64 4)
  %41 = load ptr, ptr %40, align 8
  %42 = getelementptr inbounds nuw i8, ptr %12, i64 96
  %43 = load double, ptr %42, align 8, !tbaa !6
  tail call void @__catalyst__qis__RY(double %43, ptr %41, ptr null)
  %44 = getelementptr inbounds nuw i8, ptr %12, i64 104
  %45 = load double, ptr %44, align 8, !tbaa !6
  tail call void @__catalyst__qis__RZ(double %45, ptr %41, ptr null)
  %46 = getelementptr inbounds nuw i8, ptr %12, i64 112
  %47 = load double, ptr %46, align 8, !tbaa !6
  tail call void @__catalyst__qis__RY(double %47, ptr %41, ptr null)
  %48 = getelementptr inbounds nuw i8, ptr %12, i64 120
  %49 = load double, ptr %48, align 8, !tbaa !6
  tail call void @__catalyst__qis__RZ(double %49, ptr %41, ptr null)
  %50 = tail call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %9, i64 3)
  %51 = load ptr, ptr %50, align 8
  %52 = getelementptr inbounds nuw i8, ptr %12, i64 128
  %53 = load double, ptr %52, align 8, !tbaa !6
  tail call void @__catalyst__qis__RY(double %53, ptr %51, ptr null)
  %54 = getelementptr inbounds nuw i8, ptr %12, i64 136
  %55 = load double, ptr %54, align 8, !tbaa !6
  tail call void @__catalyst__qis__RZ(double %55, ptr %51, ptr null)
  %56 = getelementptr inbounds nuw i8, ptr %12, i64 144
  %57 = load double, ptr %56, align 8, !tbaa !6
  tail call void @__catalyst__qis__RY(double %57, ptr %51, ptr null)
  %58 = getelementptr inbounds nuw i8, ptr %12, i64 152
  %59 = load double, ptr %58, align 8, !tbaa !6
  tail call void @__catalyst__qis__RZ(double %59, ptr %51, ptr null)
  %60 = tail call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %9, i64 2)
  %61 = load ptr, ptr %60, align 8
  %62 = getelementptr inbounds nuw i8, ptr %12, i64 160
  %63 = load double, ptr %62, align 8, !tbaa !6
  tail call void @__catalyst__qis__RY(double %63, ptr %61, ptr null)
  %64 = getelementptr inbounds nuw i8, ptr %12, i64 168
  %65 = load double, ptr %64, align 8, !tbaa !6
  tail call void @__catalyst__qis__RZ(double %65, ptr %61, ptr null)
  %66 = getelementptr inbounds nuw i8, ptr %12, i64 176
  %67 = load double, ptr %66, align 8, !tbaa !6
  tail call void @__catalyst__qis__RY(double %67, ptr %61, ptr null)
  %68 = getelementptr inbounds nuw i8, ptr %12, i64 184
  %69 = load double, ptr %68, align 8, !tbaa !6
  tail call void @__catalyst__qis__RZ(double %69, ptr %61, ptr null)
  %70 = tail call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %9, i64 0)
  %71 = load ptr, ptr %70, align 8
  %72 = getelementptr inbounds nuw i8, ptr %12, i64 192
  %73 = load double, ptr %72, align 8, !tbaa !6
  tail call void @__catalyst__qis__RY(double %73, ptr %71, ptr null)
  %74 = getelementptr inbounds nuw i8, ptr %12, i64 200
  %75 = load double, ptr %74, align 8, !tbaa !6
  tail call void @__catalyst__qis__RZ(double %75, ptr %71, ptr null)
  %76 = getelementptr inbounds nuw i8, ptr %12, i64 208
  %77 = load double, ptr %76, align 8, !tbaa !6
  tail call void @__catalyst__qis__RY(double %77, ptr %71, ptr null)
  %78 = getelementptr inbounds nuw i8, ptr %12, i64 216
  %79 = load double, ptr %78, align 8, !tbaa !6
  tail call void @__catalyst__qis__RZ(double %79, ptr %71, ptr null)
  %80 = tail call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %9, i64 1)
  %81 = load ptr, ptr %80, align 8
  %82 = getelementptr inbounds nuw i8, ptr %12, i64 224
  %83 = load double, ptr %82, align 8, !tbaa !6
  tail call void @__catalyst__qis__RY(double %83, ptr %81, ptr null)
  %84 = getelementptr inbounds nuw i8, ptr %12, i64 232
  %85 = load double, ptr %84, align 8, !tbaa !6
  tail call void @__catalyst__qis__RZ(double %85, ptr %81, ptr null)
  %86 = getelementptr inbounds nuw i8, ptr %12, i64 240
  %87 = load double, ptr %86, align 8, !tbaa !6
  tail call void @__catalyst__qis__RY(double %87, ptr %81, ptr null)
  %88 = getelementptr inbounds nuw i8, ptr %12, i64 248
  %89 = load double, ptr %88, align 8, !tbaa !6
  tail call void @__catalyst__qis__RZ(double %89, ptr %81, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %71, ptr %81, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %81, ptr %61, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %61, ptr %51, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %51, ptr %41, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %41, ptr %31, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %31, ptr %21, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %21, ptr %11, ptr null)
  %90 = getelementptr inbounds nuw i8, ptr %12, i64 256
  %91 = load double, ptr %90, align 8, !tbaa !6
  tail call void @__catalyst__qis__RZ(double %91, ptr %21, ptr null)
  %92 = getelementptr inbounds nuw i8, ptr %12, i64 264
  %93 = load double, ptr %92, align 8, !tbaa !6
  tail call void @__catalyst__qis__RY(double %93, ptr %21, ptr null)
  %94 = getelementptr inbounds nuw i8, ptr %12, i64 272
  %95 = load double, ptr %94, align 8, !tbaa !6
  tail call void @__catalyst__qis__RZ(double %95, ptr %21, ptr null)
  %96 = getelementptr inbounds nuw i8, ptr %12, i64 280
  %97 = load double, ptr %96, align 8, !tbaa !6
  tail call void @__catalyst__qis__RZ(double %97, ptr %41, ptr null)
  %98 = getelementptr inbounds nuw i8, ptr %12, i64 288
  %99 = load double, ptr %98, align 8, !tbaa !6
  tail call void @__catalyst__qis__RY(double %99, ptr %41, ptr null)
  %100 = getelementptr inbounds nuw i8, ptr %12, i64 296
  %101 = load double, ptr %100, align 8, !tbaa !6
  tail call void @__catalyst__qis__RZ(double %101, ptr %41, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %11, ptr %71, ptr null)
  %102 = getelementptr inbounds nuw i8, ptr %12, i64 304
  %103 = load double, ptr %102, align 8, !tbaa !6
  tail call void @__catalyst__qis__RZ(double %103, ptr %71, ptr null)
  %104 = getelementptr inbounds nuw i8, ptr %12, i64 312
  %105 = load double, ptr %104, align 8, !tbaa !6
  tail call void @__catalyst__qis__RY(double %105, ptr %71, ptr null)
  %106 = getelementptr inbounds nuw i8, ptr %12, i64 320
  %107 = load double, ptr %106, align 8, !tbaa !6
  tail call void @__catalyst__qis__RZ(double %107, ptr %71, ptr null)
  %108 = getelementptr inbounds nuw i8, ptr %12, i64 328
  %109 = load double, ptr %108, align 8, !tbaa !6
  tail call void @__catalyst__qis__RZ(double %109, ptr %61, ptr null)
  %110 = getelementptr inbounds nuw i8, ptr %12, i64 336
  %111 = load double, ptr %110, align 8, !tbaa !6
  tail call void @__catalyst__qis__RY(double %111, ptr %61, ptr null)
  %112 = getelementptr inbounds nuw i8, ptr %12, i64 344
  %113 = load double, ptr %112, align 8, !tbaa !6
  tail call void @__catalyst__qis__RZ(double %113, ptr %61, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %71, ptr %61, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %61, ptr %41, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %41, ptr %21, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %21, ptr %71, ptr null)
  %114 = getelementptr inbounds nuw i8, ptr %12, i64 352
  %115 = load double, ptr %114, align 8, !tbaa !6
  tail call void @__catalyst__qis__RZ(double %115, ptr %71, ptr null)
  %116 = getelementptr inbounds nuw i8, ptr %12, i64 360
  %117 = load double, ptr %116, align 8, !tbaa !6
  tail call void @__catalyst__qis__RY(double %117, ptr %71, ptr null)
  %118 = getelementptr inbounds nuw i8, ptr %12, i64 368
  %119 = load double, ptr %118, align 8, !tbaa !6
  tail call void @__catalyst__qis__RZ(double %119, ptr %71, ptr null)
  %120 = getelementptr inbounds nuw i8, ptr %12, i64 376
  %121 = load double, ptr %120, align 8, !tbaa !6
  tail call void @__catalyst__qis__RZ(double %121, ptr %31, ptr null)
  %122 = getelementptr inbounds nuw i8, ptr %12, i64 384
  %123 = load double, ptr %122, align 8, !tbaa !6
  tail call void @__catalyst__qis__RY(double %123, ptr %31, ptr null)
  %124 = getelementptr inbounds nuw i8, ptr %12, i64 392
  %125 = load double, ptr %124, align 8, !tbaa !6
  tail call void @__catalyst__qis__RZ(double %125, ptr %31, ptr null)
  %126 = getelementptr inbounds nuw i8, ptr %12, i64 400
  %127 = load double, ptr %126, align 8, !tbaa !6
  tail call void @__catalyst__qis__RZ(double %127, ptr %81, ptr null)
  %128 = getelementptr inbounds nuw i8, ptr %12, i64 408
  %129 = load double, ptr %128, align 8, !tbaa !6
  tail call void @__catalyst__qis__RY(double %129, ptr %81, ptr null)
  %130 = getelementptr inbounds nuw i8, ptr %12, i64 416
  %131 = load double, ptr %130, align 8, !tbaa !6
  tail call void @__catalyst__qis__RZ(double %131, ptr %81, ptr null)
  %132 = getelementptr inbounds nuw i8, ptr %12, i64 424
  %133 = load double, ptr %132, align 8, !tbaa !6
  tail call void @__catalyst__qis__RZ(double %133, ptr %51, ptr null)
  %134 = getelementptr inbounds nuw i8, ptr %12, i64 432
  %135 = load double, ptr %134, align 8, !tbaa !6
  tail call void @__catalyst__qis__RY(double %135, ptr %51, ptr null)
  %136 = getelementptr inbounds nuw i8, ptr %12, i64 440
  %137 = load double, ptr %136, align 8, !tbaa !6
  tail call void @__catalyst__qis__RZ(double %137, ptr %51, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %81, ptr %51, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %51, ptr %31, ptr null)
  %138 = getelementptr inbounds nuw i8, ptr %12, i64 448
  %139 = load double, ptr %138, align 8, !tbaa !6
  tail call void @__catalyst__qis__RZ(double %139, ptr %51, ptr null)
  %140 = getelementptr inbounds nuw i8, ptr %12, i64 456
  %141 = load double, ptr %140, align 8, !tbaa !6
  tail call void @__catalyst__qis__RY(double %141, ptr %51, ptr null)
  %142 = getelementptr inbounds nuw i8, ptr %12, i64 464
  %143 = load double, ptr %142, align 8, !tbaa !6
  tail call void @__catalyst__qis__RZ(double %143, ptr %51, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %71, ptr %51, ptr null)
  %144 = getelementptr inbounds nuw i8, ptr %12, i64 472
  %145 = load double, ptr %144, align 8, !tbaa !6
  tail call void @__catalyst__qis__RZ(double %145, ptr %61, ptr null)
  %146 = getelementptr inbounds nuw i8, ptr %12, i64 480
  %147 = load double, ptr %146, align 8, !tbaa !6
  tail call void @__catalyst__qis__RY(double %147, ptr %61, ptr null)
  %148 = getelementptr inbounds nuw i8, ptr %12, i64 488
  %149 = load double, ptr %148, align 8, !tbaa !6
  tail call void @__catalyst__qis__RZ(double %149, ptr %61, ptr null)
  %150 = getelementptr inbounds nuw i8, ptr %12, i64 496
  %151 = load double, ptr %150, align 8, !tbaa !6
  tail call void @__catalyst__qis__RZ(double %151, ptr %11, ptr null)
  %152 = getelementptr inbounds nuw i8, ptr %12, i64 504
  %153 = load double, ptr %152, align 8, !tbaa !6
  tail call void @__catalyst__qis__RY(double %153, ptr %11, ptr null)
  %154 = getelementptr inbounds nuw i8, ptr %12, i64 512
  %155 = load double, ptr %154, align 8, !tbaa !6
  tail call void @__catalyst__qis__RZ(double %155, ptr %11, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %31, ptr %11, ptr null)
  %156 = getelementptr inbounds nuw i8, ptr %12, i64 520
  %157 = load double, ptr %156, align 8, !tbaa !6
  tail call void @__catalyst__qis__RZ(double %157, ptr %31, ptr null)
  %158 = getelementptr inbounds nuw i8, ptr %12, i64 528
  %159 = load double, ptr %158, align 8, !tbaa !6
  tail call void @__catalyst__qis__RY(double %159, ptr %31, ptr null)
  %160 = getelementptr inbounds nuw i8, ptr %12, i64 536
  %161 = load double, ptr %160, align 8, !tbaa !6
  tail call void @__catalyst__qis__RZ(double %161, ptr %31, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %61, ptr %31, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %31, ptr %71, ptr null)
  %162 = getelementptr inbounds nuw i8, ptr %12, i64 544
  %163 = load double, ptr %162, align 8, !tbaa !6
  tail call void @__catalyst__qis__RZ(double %163, ptr %71, ptr null)
  %164 = getelementptr inbounds nuw i8, ptr %12, i64 552
  %165 = load double, ptr %164, align 8, !tbaa !6
  tail call void @__catalyst__qis__RY(double %165, ptr %71, ptr null)
  %166 = getelementptr inbounds nuw i8, ptr %12, i64 560
  %167 = load double, ptr %166, align 8, !tbaa !6
  tail call void @__catalyst__qis__RZ(double %167, ptr %71, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %11, ptr %81, ptr null)
  %168 = getelementptr inbounds nuw i8, ptr %12, i64 568
  %169 = load double, ptr %168, align 8, !tbaa !6
  tail call void @__catalyst__qis__RZ(double %169, ptr %11, ptr null)
  %170 = getelementptr inbounds nuw i8, ptr %12, i64 576
  %171 = load double, ptr %170, align 8, !tbaa !6
  tail call void @__catalyst__qis__RY(double %171, ptr %11, ptr null)
  %172 = getelementptr inbounds nuw i8, ptr %12, i64 584
  %173 = load double, ptr %172, align 8, !tbaa !6
  tail call void @__catalyst__qis__RZ(double %173, ptr %11, ptr null)
  %174 = getelementptr inbounds nuw i8, ptr %12, i64 592
  %175 = load double, ptr %174, align 8, !tbaa !6
  tail call void @__catalyst__qis__RZ(double %175, ptr %81, ptr null)
  %176 = getelementptr inbounds nuw i8, ptr %12, i64 600
  %177 = load double, ptr %176, align 8, !tbaa !6
  tail call void @__catalyst__qis__RY(double %177, ptr %81, ptr null)
  %178 = getelementptr inbounds nuw i8, ptr %12, i64 608
  %179 = load double, ptr %178, align 8, !tbaa !6
  tail call void @__catalyst__qis__RZ(double %179, ptr %81, ptr null)
  %180 = getelementptr inbounds nuw i8, ptr %12, i64 616
  %181 = load double, ptr %180, align 8, !tbaa !6
  tail call void @__catalyst__qis__RZ(double %181, ptr %41, ptr null)
  %182 = getelementptr inbounds nuw i8, ptr %12, i64 624
  %183 = load double, ptr %182, align 8, !tbaa !6
  tail call void @__catalyst__qis__RY(double %183, ptr %41, ptr null)
  %184 = getelementptr inbounds nuw i8, ptr %12, i64 632
  %185 = load double, ptr %184, align 8, !tbaa !6
  tail call void @__catalyst__qis__RZ(double %185, ptr %41, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %81, ptr %41, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %41, ptr %11, ptr null)
  %186 = getelementptr inbounds nuw i8, ptr %12, i64 640
  %187 = load double, ptr %186, align 8, !tbaa !6
  tail call void @__catalyst__qis__RZ(double %187, ptr %41, ptr null)
  %188 = getelementptr inbounds nuw i8, ptr %12, i64 648
  %189 = load double, ptr %188, align 8, !tbaa !6
  tail call void @__catalyst__qis__RY(double %189, ptr %41, ptr null)
  %190 = getelementptr inbounds nuw i8, ptr %12, i64 656
  %191 = load double, ptr %190, align 8, !tbaa !6
  tail call void @__catalyst__qis__RZ(double %191, ptr %41, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %71, ptr %41, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %41, ptr %71, ptr null)
  %192 = getelementptr inbounds nuw i8, ptr %12, i64 664
  %193 = load double, ptr %192, align 8, !tbaa !6
  tail call void @__catalyst__qis__RZ(double %193, ptr %21, ptr null)
  %194 = getelementptr inbounds nuw i8, ptr %12, i64 672
  %195 = load double, ptr %194, align 8, !tbaa !6
  tail call void @__catalyst__qis__RY(double %195, ptr %21, ptr null)
  %196 = getelementptr inbounds nuw i8, ptr %12, i64 680
  %197 = load double, ptr %196, align 8, !tbaa !6
  tail call void @__catalyst__qis__RZ(double %197, ptr %21, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %51, ptr %21, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %21, ptr %81, ptr null)
  %198 = getelementptr inbounds nuw i8, ptr %12, i64 688
  %199 = load double, ptr %198, align 8, !tbaa !6
  tail call void @__catalyst__qis__RZ(double %199, ptr %81, ptr null)
  %200 = getelementptr inbounds nuw i8, ptr %12, i64 696
  %201 = load double, ptr %200, align 8, !tbaa !6
  tail call void @__catalyst__qis__RY(double %201, ptr %81, ptr null)
  %202 = getelementptr inbounds nuw i8, ptr %12, i64 704
  %203 = load double, ptr %202, align 8, !tbaa !6
  tail call void @__catalyst__qis__RZ(double %203, ptr %81, ptr null)
  %204 = getelementptr inbounds nuw i8, ptr %12, i64 712
  %205 = load double, ptr %204, align 8, !tbaa !6
  tail call void @__catalyst__qis__RZ(double %205, ptr %31, ptr null)
  %206 = getelementptr inbounds nuw i8, ptr %12, i64 720
  %207 = load double, ptr %206, align 8, !tbaa !6
  tail call void @__catalyst__qis__RY(double %207, ptr %31, ptr null)
  %208 = getelementptr inbounds nuw i8, ptr %12, i64 728
  %209 = load double, ptr %208, align 8, !tbaa !6
  tail call void @__catalyst__qis__RZ(double %209, ptr %31, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %81, ptr %31, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %31, ptr %81, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %11, ptr %61, ptr null)
  %210 = getelementptr inbounds nuw i8, ptr %12, i64 736
  %211 = load double, ptr %210, align 8, !tbaa !6
  tail call void @__catalyst__qis__RZ(double %211, ptr %61, ptr null)
  %212 = getelementptr inbounds nuw i8, ptr %12, i64 744
  %213 = load double, ptr %212, align 8, !tbaa !6
  tail call void @__catalyst__qis__RY(double %213, ptr %61, ptr null)
  %214 = getelementptr inbounds nuw i8, ptr %12, i64 752
  %215 = load double, ptr %214, align 8, !tbaa !6
  tail call void @__catalyst__qis__RZ(double %215, ptr %61, ptr null)
  %216 = getelementptr inbounds nuw i8, ptr %12, i64 760
  %217 = load double, ptr %216, align 8, !tbaa !6
  tail call void @__catalyst__qis__RZ(double %217, ptr %21, ptr null)
  %218 = getelementptr inbounds nuw i8, ptr %12, i64 768
  %219 = load double, ptr %218, align 8, !tbaa !6
  tail call void @__catalyst__qis__RY(double %219, ptr %21, ptr null)
  %220 = getelementptr inbounds nuw i8, ptr %12, i64 776
  %221 = load double, ptr %220, align 8, !tbaa !6
  tail call void @__catalyst__qis__RZ(double %221, ptr %21, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %61, ptr %21, ptr null)
  tail call void @__catalyst__qis__CNOT(ptr %21, ptr %61, ptr null)
  %222 = getelementptr inbounds nuw i8, ptr %12, i64 784
  %223 = load double, ptr %222, align 8, !tbaa !6
  tail call void @__catalyst__qis__RZ(double %223, ptr %51, ptr null)
  %224 = getelementptr inbounds nuw i8, ptr %12, i64 792
  %225 = load double, ptr %224, align 8, !tbaa !6
  tail call void @__catalyst__qis__RY(double %225, ptr %51, ptr null)
  %226 = getelementptr inbounds nuw i8, ptr %12, i64 800
  %227 = load double, ptr %226, align 8, !tbaa !6
  tail call void @__catalyst__qis__RZ(double %227, ptr %51, ptr null)
  %228 = getelementptr inbounds nuw i8, ptr %12, i64 808
  %229 = load double, ptr %228, align 8, !tbaa !6
  tail call void @__catalyst__qis__RZ(double %229, ptr %11, ptr null)
  %230 = getelementptr inbounds nuw i8, ptr %12, i64 816
  %231 = load double, ptr %230, align 8, !tbaa !6
  tail call void @__catalyst__qis__RY(double %231, ptr %11, ptr null)
  %232 = getelementptr inbounds nuw i8, ptr %12, i64 824
  %233 = load double, ptr %232, align 8, !tbaa !6
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
  store double %235, ptr %240, align 64, !tbaa !6
  tail call void @__catalyst__rt__qubit_release_array(ptr %9)
  tail call void @__catalyst__rt__device_release()
  %241 = load double, ptr %240, align 64, !tbaa !6
  %242 = extractvalue { ptr, ptr, i64 } %8, 1
  store double %241, ptr %242, align 8, !tbaa !6
  ret void
}

define noalias noundef ptr @qnode_forward_0.quantum.augfwd(ptr %0, ptr readnone captures(none) %1, ptr %2, ptr readnone captures(none) %3, ptr %4, ptr readnone captures(none) %5, ptr %6, ptr readnone captures(none) %7) {
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
  %21 = load float, ptr %20, align 4, !tbaa !4
  %22 = getelementptr inbounds nuw i8, ptr %1, i64 292
  %23 = load float, ptr %22, align 4, !tbaa !4
  %24 = getelementptr inbounds nuw i8, ptr %1, i64 288
  %25 = load float, ptr %24, align 4, !tbaa !4
  %26 = getelementptr inbounds nuw i8, ptr %1, i64 200
  %27 = load float, ptr %26, align 4, !tbaa !4
  %28 = getelementptr inbounds nuw i8, ptr %1, i64 196
  %29 = load float, ptr %28, align 4, !tbaa !4
  %30 = getelementptr inbounds nuw i8, ptr %1, i64 192
  %31 = load float, ptr %30, align 4, !tbaa !4
  %32 = getelementptr inbounds nuw i8, ptr %1, i64 176
  %33 = load float, ptr %32, align 4, !tbaa !4
  %34 = getelementptr inbounds nuw i8, ptr %1, i64 172
  %35 = load float, ptr %34, align 4, !tbaa !4
  %36 = getelementptr inbounds nuw i8, ptr %1, i64 168
  %37 = load float, ptr %36, align 4, !tbaa !4
  %38 = getelementptr inbounds nuw i8, ptr %1, i64 92
  %39 = load float, ptr %38, align 4, !tbaa !4
  %40 = getelementptr inbounds nuw i8, ptr %1, i64 88
  %41 = load float, ptr %40, align 4, !tbaa !4
  %42 = getelementptr inbounds nuw i8, ptr %1, i64 84
  %43 = load float, ptr %42, align 4, !tbaa !4
  %44 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 96)
  %45 = ptrtoint ptr %44 to i64
  %46 = add i64 %45, 63
  %47 = and i64 %46, -64
  %48 = inttoptr i64 %47 to ptr
  store float 0x400921FB60000000, ptr %48, align 64, !tbaa !4
  %49 = getelementptr inbounds nuw i8, ptr %48, i64 4
  store float 0x400921FB60000000, ptr %49, align 4, !tbaa !4
  %50 = getelementptr inbounds nuw i8, ptr %48, i64 8
  store float 0x400921FB60000000, ptr %50, align 8, !tbaa !4
  %51 = getelementptr inbounds nuw i8, ptr %48, i64 12
  store float 0x400921FB60000000, ptr %51, align 4, !tbaa !4
  %52 = getelementptr inbounds nuw i8, ptr %48, i64 16
  store float 0x400921FB60000000, ptr %52, align 16, !tbaa !4
  %53 = getelementptr inbounds nuw i8, ptr %48, i64 20
  store float 0x400921FB60000000, ptr %53, align 4, !tbaa !4
  %54 = getelementptr inbounds nuw i8, ptr %48, i64 24
  store float 0x400921FB60000000, ptr %54, align 8, !tbaa !4
  %55 = getelementptr inbounds nuw i8, ptr %48, i64 28
  store float 0x400921FB60000000, ptr %55, align 4, !tbaa !4
  %56 = load float, ptr %10, align 4, !tbaa !4
  %57 = fmul float %56, 0x400921FB60000000
  store float %57, ptr %48, align 64, !tbaa !4
  %58 = getelementptr inbounds nuw i8, ptr %10, i64 4
  %59 = load float, ptr %58, align 4, !tbaa !4
  %60 = fmul float %59, 0x400921FB60000000
  store float %60, ptr %49, align 4, !tbaa !4
  %61 = getelementptr inbounds nuw i8, ptr %10, i64 8
  %62 = load float, ptr %61, align 4, !tbaa !4
  %63 = fmul float %62, 0x400921FB60000000
  store float %63, ptr %50, align 8, !tbaa !4
  %64 = getelementptr inbounds nuw i8, ptr %10, i64 12
  %65 = load float, ptr %64, align 4, !tbaa !4
  %66 = fmul float %65, 0x400921FB60000000
  store float %66, ptr %51, align 4, !tbaa !4
  %67 = getelementptr inbounds nuw i8, ptr %10, i64 16
  %68 = load float, ptr %67, align 4, !tbaa !4
  %69 = fmul float %68, 0x400921FB60000000
  store float %69, ptr %52, align 16, !tbaa !4
  %70 = getelementptr inbounds nuw i8, ptr %10, i64 20
  %71 = load float, ptr %70, align 4, !tbaa !4
  %72 = fmul float %71, 0x400921FB60000000
  store float %72, ptr %53, align 4, !tbaa !4
  %73 = getelementptr inbounds nuw i8, ptr %10, i64 24
  %74 = load float, ptr %73, align 4, !tbaa !4
  %75 = fmul float %74, 0x400921FB60000000
  store float %75, ptr %54, align 8, !tbaa !4
  %76 = getelementptr inbounds nuw i8, ptr %10, i64 28
  %77 = load float, ptr %76, align 4, !tbaa !4
  %78 = fmul float %77, 0x400921FB60000000
  store float %78, ptr %55, align 4, !tbaa !4
  %79 = fpext float %78 to double
  store double %79, ptr %19, align 8, !tbaa !6
  %80 = fpext float %43 to double
  %81 = getelementptr inbounds nuw i8, ptr %19, i64 8
  store double %80, ptr %81, align 8, !tbaa !6
  %82 = fpext float %41 to double
  %83 = getelementptr inbounds nuw i8, ptr %19, i64 16
  store double %82, ptr %83, align 8, !tbaa !6
  %84 = fpext float %39 to double
  %85 = getelementptr inbounds nuw i8, ptr %19, i64 24
  store double %84, ptr %85, align 8, !tbaa !6
  %86 = getelementptr inbounds nuw i8, ptr %1, i64 80
  %87 = load float, ptr %86, align 4, !tbaa !4
  %88 = getelementptr inbounds nuw i8, ptr %1, i64 76
  %89 = load float, ptr %88, align 4, !tbaa !4
  %90 = getelementptr inbounds nuw i8, ptr %1, i64 72
  %91 = load float, ptr %90, align 4, !tbaa !4
  %92 = fpext float %75 to double
  %93 = getelementptr inbounds nuw i8, ptr %19, i64 32
  store double %92, ptr %93, align 8, !tbaa !6
  %94 = fpext float %91 to double
  %95 = getelementptr inbounds nuw i8, ptr %19, i64 40
  store double %94, ptr %95, align 8, !tbaa !6
  %96 = fpext float %89 to double
  %97 = getelementptr inbounds nuw i8, ptr %19, i64 48
  store double %96, ptr %97, align 8, !tbaa !6
  %98 = fpext float %87 to double
  %99 = getelementptr inbounds nuw i8, ptr %19, i64 56
  store double %98, ptr %99, align 8, !tbaa !6
  %100 = getelementptr inbounds nuw i8, ptr %1, i64 68
  %101 = load float, ptr %100, align 4, !tbaa !4
  %102 = getelementptr inbounds nuw i8, ptr %1, i64 64
  %103 = load float, ptr %102, align 4, !tbaa !4
  %104 = getelementptr inbounds nuw i8, ptr %1, i64 60
  %105 = load float, ptr %104, align 4, !tbaa !4
  %106 = fpext float %72 to double
  %107 = getelementptr inbounds nuw i8, ptr %19, i64 64
  store double %106, ptr %107, align 8, !tbaa !6
  %108 = fpext float %105 to double
  %109 = getelementptr inbounds nuw i8, ptr %19, i64 72
  store double %108, ptr %109, align 8, !tbaa !6
  %110 = fpext float %103 to double
  %111 = getelementptr inbounds nuw i8, ptr %19, i64 80
  store double %110, ptr %111, align 8, !tbaa !6
  %112 = fpext float %101 to double
  %113 = getelementptr inbounds nuw i8, ptr %19, i64 88
  store double %112, ptr %113, align 8, !tbaa !6
  %114 = getelementptr inbounds nuw i8, ptr %1, i64 56
  %115 = load float, ptr %114, align 4, !tbaa !4
  %116 = getelementptr inbounds nuw i8, ptr %1, i64 52
  %117 = load float, ptr %116, align 4, !tbaa !4
  %118 = getelementptr inbounds nuw i8, ptr %1, i64 48
  %119 = load float, ptr %118, align 4, !tbaa !4
  %120 = fpext float %69 to double
  %121 = getelementptr inbounds nuw i8, ptr %19, i64 96
  store double %120, ptr %121, align 8, !tbaa !6
  %122 = fpext float %119 to double
  %123 = getelementptr inbounds nuw i8, ptr %19, i64 104
  store double %122, ptr %123, align 8, !tbaa !6
  %124 = fpext float %117 to double
  %125 = getelementptr inbounds nuw i8, ptr %19, i64 112
  store double %124, ptr %125, align 8, !tbaa !6
  %126 = fpext float %115 to double
  %127 = getelementptr inbounds nuw i8, ptr %19, i64 120
  store double %126, ptr %127, align 8, !tbaa !6
  %128 = getelementptr inbounds nuw i8, ptr %1, i64 44
  %129 = load float, ptr %128, align 4, !tbaa !4
  %130 = getelementptr inbounds nuw i8, ptr %1, i64 40
  %131 = load float, ptr %130, align 4, !tbaa !4
  %132 = getelementptr inbounds nuw i8, ptr %1, i64 36
  %133 = load float, ptr %132, align 4, !tbaa !4
  %134 = fpext float %66 to double
  %135 = getelementptr inbounds nuw i8, ptr %19, i64 128
  store double %134, ptr %135, align 8, !tbaa !6
  %136 = fpext float %133 to double
  %137 = getelementptr inbounds nuw i8, ptr %19, i64 136
  store double %136, ptr %137, align 8, !tbaa !6
  %138 = fpext float %131 to double
  %139 = getelementptr inbounds nuw i8, ptr %19, i64 144
  store double %138, ptr %139, align 8, !tbaa !6
  %140 = fpext float %129 to double
  %141 = getelementptr inbounds nuw i8, ptr %19, i64 152
  store double %140, ptr %141, align 8, !tbaa !6
  %142 = getelementptr inbounds nuw i8, ptr %1, i64 32
  %143 = load float, ptr %142, align 4, !tbaa !4
  %144 = getelementptr inbounds nuw i8, ptr %1, i64 28
  %145 = load float, ptr %144, align 4, !tbaa !4
  %146 = getelementptr inbounds nuw i8, ptr %1, i64 24
  %147 = load float, ptr %146, align 4, !tbaa !4
  %148 = load float, ptr %50, align 8, !tbaa !4
  %149 = fpext float %148 to double
  %150 = getelementptr inbounds nuw i8, ptr %19, i64 160
  store double %149, ptr %150, align 8, !tbaa !6
  %151 = fpext float %147 to double
  %152 = getelementptr inbounds nuw i8, ptr %19, i64 168
  store double %151, ptr %152, align 8, !tbaa !6
  %153 = fpext float %145 to double
  %154 = getelementptr inbounds nuw i8, ptr %19, i64 176
  store double %153, ptr %154, align 8, !tbaa !6
  %155 = fpext float %143 to double
  %156 = getelementptr inbounds nuw i8, ptr %19, i64 184
  store double %155, ptr %156, align 8, !tbaa !6
  %157 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %158 = load float, ptr %157, align 4, !tbaa !4
  %159 = getelementptr inbounds nuw i8, ptr %1, i64 4
  %160 = load float, ptr %159, align 4, !tbaa !4
  %161 = load float, ptr %1, align 4, !tbaa !4
  %162 = load float, ptr %48, align 64, !tbaa !4
  %163 = fpext float %162 to double
  %164 = getelementptr inbounds nuw i8, ptr %19, i64 192
  store double %163, ptr %164, align 8, !tbaa !6
  %165 = fpext float %161 to double
  %166 = getelementptr inbounds nuw i8, ptr %19, i64 200
  store double %165, ptr %166, align 8, !tbaa !6
  %167 = fpext float %160 to double
  %168 = getelementptr inbounds nuw i8, ptr %19, i64 208
  store double %167, ptr %168, align 8, !tbaa !6
  %169 = fpext float %158 to double
  %170 = getelementptr inbounds nuw i8, ptr %19, i64 216
  store double %169, ptr %170, align 8, !tbaa !6
  %171 = getelementptr inbounds nuw i8, ptr %1, i64 20
  %172 = load float, ptr %171, align 4, !tbaa !4
  %173 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %174 = load float, ptr %173, align 4, !tbaa !4
  %175 = getelementptr inbounds nuw i8, ptr %1, i64 12
  %176 = load float, ptr %175, align 4, !tbaa !4
  %177 = load float, ptr %49, align 4, !tbaa !4
  tail call void @_mlir_memref_to_llvm_free(ptr %44)
  %178 = fpext float %177 to double
  %179 = getelementptr inbounds nuw i8, ptr %19, i64 224
  store double %178, ptr %179, align 8, !tbaa !6
  %180 = fpext float %176 to double
  %181 = getelementptr inbounds nuw i8, ptr %19, i64 232
  store double %180, ptr %181, align 8, !tbaa !6
  %182 = fpext float %174 to double
  %183 = getelementptr inbounds nuw i8, ptr %19, i64 240
  store double %182, ptr %183, align 8, !tbaa !6
  %184 = fpext float %172 to double
  %185 = getelementptr inbounds nuw i8, ptr %19, i64 248
  store double %184, ptr %185, align 8, !tbaa !6
  %186 = fpext float %37 to double
  %187 = getelementptr inbounds nuw i8, ptr %19, i64 256
  store double %186, ptr %187, align 8, !tbaa !6
  %188 = fpext float %35 to double
  %189 = getelementptr inbounds nuw i8, ptr %19, i64 264
  store double %188, ptr %189, align 8, !tbaa !6
  %190 = fpext float %33 to double
  %191 = getelementptr inbounds nuw i8, ptr %19, i64 272
  store double %190, ptr %191, align 8, !tbaa !6
  %192 = getelementptr inbounds nuw i8, ptr %1, i64 152
  %193 = load float, ptr %192, align 4, !tbaa !4
  %194 = getelementptr inbounds nuw i8, ptr %1, i64 148
  %195 = load float, ptr %194, align 4, !tbaa !4
  %196 = getelementptr inbounds nuw i8, ptr %1, i64 144
  %197 = load float, ptr %196, align 4, !tbaa !4
  %198 = fpext float %197 to double
  %199 = getelementptr inbounds nuw i8, ptr %19, i64 280
  store double %198, ptr %199, align 8, !tbaa !6
  %200 = fpext float %195 to double
  %201 = getelementptr inbounds nuw i8, ptr %19, i64 288
  store double %200, ptr %201, align 8, !tbaa !6
  %202 = fpext float %193 to double
  %203 = getelementptr inbounds nuw i8, ptr %19, i64 296
  store double %202, ptr %203, align 8, !tbaa !6
  %204 = getelementptr inbounds nuw i8, ptr %1, i64 104
  %205 = load float, ptr %204, align 4, !tbaa !4
  %206 = getelementptr inbounds nuw i8, ptr %1, i64 100
  %207 = load float, ptr %206, align 4, !tbaa !4
  %208 = getelementptr inbounds nuw i8, ptr %1, i64 96
  %209 = load float, ptr %208, align 4, !tbaa !4
  %210 = fpext float %209 to double
  %211 = getelementptr inbounds nuw i8, ptr %19, i64 304
  store double %210, ptr %211, align 8, !tbaa !6
  %212 = fpext float %207 to double
  %213 = getelementptr inbounds nuw i8, ptr %19, i64 312
  store double %212, ptr %213, align 8, !tbaa !6
  %214 = fpext float %205 to double
  %215 = getelementptr inbounds nuw i8, ptr %19, i64 320
  store double %214, ptr %215, align 8, !tbaa !6
  %216 = getelementptr inbounds nuw i8, ptr %1, i64 128
  %217 = load float, ptr %216, align 4, !tbaa !4
  %218 = getelementptr inbounds nuw i8, ptr %1, i64 124
  %219 = load float, ptr %218, align 4, !tbaa !4
  %220 = getelementptr inbounds nuw i8, ptr %1, i64 120
  %221 = load float, ptr %220, align 4, !tbaa !4
  %222 = fpext float %221 to double
  %223 = getelementptr inbounds nuw i8, ptr %19, i64 328
  store double %222, ptr %223, align 8, !tbaa !6
  %224 = fpext float %219 to double
  %225 = getelementptr inbounds nuw i8, ptr %19, i64 336
  store double %224, ptr %225, align 8, !tbaa !6
  %226 = fpext float %217 to double
  %227 = getelementptr inbounds nuw i8, ptr %19, i64 344
  store double %226, ptr %227, align 8, !tbaa !6
  %228 = fpext float %31 to double
  %229 = getelementptr inbounds nuw i8, ptr %19, i64 352
  store double %228, ptr %229, align 8, !tbaa !6
  %230 = fpext float %29 to double
  %231 = getelementptr inbounds nuw i8, ptr %19, i64 360
  store double %230, ptr %231, align 8, !tbaa !6
  %232 = fpext float %27 to double
  %233 = getelementptr inbounds nuw i8, ptr %19, i64 368
  store double %232, ptr %233, align 8, !tbaa !6
  %234 = getelementptr inbounds nuw i8, ptr %1, i64 236
  %235 = load float, ptr %234, align 4, !tbaa !4
  %236 = getelementptr inbounds nuw i8, ptr %1, i64 232
  %237 = load float, ptr %236, align 4, !tbaa !4
  %238 = getelementptr inbounds nuw i8, ptr %1, i64 228
  %239 = load float, ptr %238, align 4, !tbaa !4
  %240 = getelementptr inbounds nuw i8, ptr %1, i64 164
  %241 = load float, ptr %240, align 4, !tbaa !4
  %242 = getelementptr inbounds nuw i8, ptr %1, i64 160
  %243 = load float, ptr %242, align 4, !tbaa !4
  %244 = getelementptr inbounds nuw i8, ptr %1, i64 156
  %245 = load float, ptr %244, align 4, !tbaa !4
  %246 = fpext float %245 to double
  %247 = getelementptr inbounds nuw i8, ptr %19, i64 376
  store double %246, ptr %247, align 8, !tbaa !6
  %248 = fpext float %243 to double
  %249 = getelementptr inbounds nuw i8, ptr %19, i64 384
  store double %248, ptr %249, align 8, !tbaa !6
  %250 = fpext float %241 to double
  %251 = getelementptr inbounds nuw i8, ptr %19, i64 392
  store double %250, ptr %251, align 8, !tbaa !6
  %252 = getelementptr inbounds nuw i8, ptr %1, i64 116
  %253 = load float, ptr %252, align 4, !tbaa !4
  %254 = getelementptr inbounds nuw i8, ptr %1, i64 112
  %255 = load float, ptr %254, align 4, !tbaa !4
  %256 = getelementptr inbounds nuw i8, ptr %1, i64 108
  %257 = load float, ptr %256, align 4, !tbaa !4
  %258 = fpext float %257 to double
  %259 = getelementptr inbounds nuw i8, ptr %19, i64 400
  store double %258, ptr %259, align 8, !tbaa !6
  %260 = fpext float %255 to double
  %261 = getelementptr inbounds nuw i8, ptr %19, i64 408
  store double %260, ptr %261, align 8, !tbaa !6
  %262 = fpext float %253 to double
  %263 = getelementptr inbounds nuw i8, ptr %19, i64 416
  store double %262, ptr %263, align 8, !tbaa !6
  %264 = getelementptr inbounds nuw i8, ptr %1, i64 140
  %265 = load float, ptr %264, align 4, !tbaa !4
  %266 = getelementptr inbounds nuw i8, ptr %1, i64 136
  %267 = load float, ptr %266, align 4, !tbaa !4
  %268 = getelementptr inbounds nuw i8, ptr %1, i64 132
  %269 = load float, ptr %268, align 4, !tbaa !4
  %270 = fpext float %269 to double
  %271 = getelementptr inbounds nuw i8, ptr %19, i64 424
  store double %270, ptr %271, align 8, !tbaa !6
  %272 = fpext float %267 to double
  %273 = getelementptr inbounds nuw i8, ptr %19, i64 432
  store double %272, ptr %273, align 8, !tbaa !6
  %274 = fpext float %265 to double
  %275 = getelementptr inbounds nuw i8, ptr %19, i64 440
  store double %274, ptr %275, align 8, !tbaa !6
  %276 = fpext float %239 to double
  %277 = getelementptr inbounds nuw i8, ptr %19, i64 448
  store double %276, ptr %277, align 8, !tbaa !6
  %278 = fpext float %237 to double
  %279 = getelementptr inbounds nuw i8, ptr %19, i64 456
  store double %278, ptr %279, align 8, !tbaa !6
  %280 = fpext float %235 to double
  %281 = getelementptr inbounds nuw i8, ptr %19, i64 464
  store double %280, ptr %281, align 8, !tbaa !6
  %282 = getelementptr inbounds nuw i8, ptr %1, i64 224
  %283 = load float, ptr %282, align 4, !tbaa !4
  %284 = getelementptr inbounds nuw i8, ptr %1, i64 220
  %285 = load float, ptr %284, align 4, !tbaa !4
  %286 = getelementptr inbounds nuw i8, ptr %1, i64 216
  %287 = load float, ptr %286, align 4, !tbaa !4
  %288 = fpext float %287 to double
  %289 = getelementptr inbounds nuw i8, ptr %19, i64 472
  store double %288, ptr %289, align 8, !tbaa !6
  %290 = fpext float %285 to double
  %291 = getelementptr inbounds nuw i8, ptr %19, i64 480
  store double %290, ptr %291, align 8, !tbaa !6
  %292 = fpext float %283 to double
  %293 = getelementptr inbounds nuw i8, ptr %19, i64 488
  store double %292, ptr %293, align 8, !tbaa !6
  %294 = getelementptr inbounds nuw i8, ptr %1, i64 260
  %295 = load float, ptr %294, align 4, !tbaa !4
  %296 = getelementptr inbounds nuw i8, ptr %1, i64 256
  %297 = load float, ptr %296, align 4, !tbaa !4
  %298 = getelementptr inbounds nuw i8, ptr %1, i64 252
  %299 = load float, ptr %298, align 4, !tbaa !4
  %300 = getelementptr inbounds nuw i8, ptr %1, i64 188
  %301 = load float, ptr %300, align 4, !tbaa !4
  %302 = getelementptr inbounds nuw i8, ptr %1, i64 184
  %303 = load float, ptr %302, align 4, !tbaa !4
  %304 = getelementptr inbounds nuw i8, ptr %1, i64 180
  %305 = load float, ptr %304, align 4, !tbaa !4
  %306 = fpext float %305 to double
  %307 = getelementptr inbounds nuw i8, ptr %19, i64 496
  store double %306, ptr %307, align 8, !tbaa !6
  %308 = fpext float %303 to double
  %309 = getelementptr inbounds nuw i8, ptr %19, i64 504
  store double %308, ptr %309, align 8, !tbaa !6
  %310 = fpext float %301 to double
  %311 = getelementptr inbounds nuw i8, ptr %19, i64 512
  store double %310, ptr %311, align 8, !tbaa !6
  %312 = fpext float %299 to double
  %313 = getelementptr inbounds nuw i8, ptr %19, i64 520
  store double %312, ptr %313, align 8, !tbaa !6
  %314 = fpext float %297 to double
  %315 = getelementptr inbounds nuw i8, ptr %19, i64 528
  store double %314, ptr %315, align 8, !tbaa !6
  %316 = fpext float %295 to double
  %317 = getelementptr inbounds nuw i8, ptr %19, i64 536
  store double %316, ptr %317, align 8, !tbaa !6
  %318 = fpext float %25 to double
  %319 = getelementptr inbounds nuw i8, ptr %19, i64 544
  store double %318, ptr %319, align 8, !tbaa !6
  %320 = fpext float %23 to double
  %321 = getelementptr inbounds nuw i8, ptr %19, i64 552
  store double %320, ptr %321, align 8, !tbaa !6
  %322 = fpext float %21 to double
  %323 = getelementptr inbounds nuw i8, ptr %19, i64 560
  store double %322, ptr %323, align 8, !tbaa !6
  %324 = getelementptr inbounds nuw i8, ptr %1, i64 344
  %325 = load float, ptr %324, align 4, !tbaa !4
  %326 = getelementptr inbounds nuw i8, ptr %1, i64 340
  %327 = load float, ptr %326, align 4, !tbaa !4
  %328 = getelementptr inbounds nuw i8, ptr %1, i64 336
  %329 = load float, ptr %328, align 4, !tbaa !4
  %330 = getelementptr inbounds nuw i8, ptr %1, i64 284
  %331 = load float, ptr %330, align 4, !tbaa !4
  %332 = getelementptr inbounds nuw i8, ptr %1, i64 280
  %333 = load float, ptr %332, align 4, !tbaa !4
  %334 = getelementptr inbounds nuw i8, ptr %1, i64 276
  %335 = load float, ptr %334, align 4, !tbaa !4
  %336 = fpext float %335 to double
  %337 = getelementptr inbounds nuw i8, ptr %19, i64 568
  store double %336, ptr %337, align 8, !tbaa !6
  %338 = fpext float %333 to double
  %339 = getelementptr inbounds nuw i8, ptr %19, i64 576
  store double %338, ptr %339, align 8, !tbaa !6
  %340 = fpext float %331 to double
  %341 = getelementptr inbounds nuw i8, ptr %19, i64 584
  store double %340, ptr %341, align 8, !tbaa !6
  %342 = getelementptr inbounds nuw i8, ptr %1, i64 212
  %343 = load float, ptr %342, align 4, !tbaa !4
  %344 = getelementptr inbounds nuw i8, ptr %1, i64 208
  %345 = load float, ptr %344, align 4, !tbaa !4
  %346 = getelementptr inbounds nuw i8, ptr %1, i64 204
  %347 = load float, ptr %346, align 4, !tbaa !4
  %348 = fpext float %347 to double
  %349 = getelementptr inbounds nuw i8, ptr %19, i64 592
  store double %348, ptr %349, align 8, !tbaa !6
  %350 = fpext float %345 to double
  %351 = getelementptr inbounds nuw i8, ptr %19, i64 600
  store double %350, ptr %351, align 8, !tbaa !6
  %352 = fpext float %343 to double
  %353 = getelementptr inbounds nuw i8, ptr %19, i64 608
  store double %352, ptr %353, align 8, !tbaa !6
  %354 = getelementptr inbounds nuw i8, ptr %1, i64 248
  %355 = load float, ptr %354, align 4, !tbaa !4
  %356 = getelementptr inbounds nuw i8, ptr %1, i64 244
  %357 = load float, ptr %356, align 4, !tbaa !4
  %358 = getelementptr inbounds nuw i8, ptr %1, i64 240
  %359 = load float, ptr %358, align 4, !tbaa !4
  %360 = fpext float %359 to double
  %361 = getelementptr inbounds nuw i8, ptr %19, i64 616
  store double %360, ptr %361, align 8, !tbaa !6
  %362 = fpext float %357 to double
  %363 = getelementptr inbounds nuw i8, ptr %19, i64 624
  store double %362, ptr %363, align 8, !tbaa !6
  %364 = fpext float %355 to double
  %365 = getelementptr inbounds nuw i8, ptr %19, i64 632
  store double %364, ptr %365, align 8, !tbaa !6
  %366 = fpext float %329 to double
  %367 = getelementptr inbounds nuw i8, ptr %19, i64 640
  store double %366, ptr %367, align 8, !tbaa !6
  %368 = fpext float %327 to double
  %369 = getelementptr inbounds nuw i8, ptr %19, i64 648
  store double %368, ptr %369, align 8, !tbaa !6
  %370 = fpext float %325 to double
  %371 = getelementptr inbounds nuw i8, ptr %19, i64 656
  store double %370, ptr %371, align 8, !tbaa !6
  %372 = getelementptr inbounds nuw i8, ptr %1, i64 308
  %373 = load float, ptr %372, align 4, !tbaa !4
  %374 = getelementptr inbounds nuw i8, ptr %1, i64 304
  %375 = load float, ptr %374, align 4, !tbaa !4
  %376 = getelementptr inbounds nuw i8, ptr %1, i64 300
  %377 = load float, ptr %376, align 4, !tbaa !4
  %378 = getelementptr inbounds nuw i8, ptr %1, i64 272
  %379 = load float, ptr %378, align 4, !tbaa !4
  %380 = getelementptr inbounds nuw i8, ptr %1, i64 268
  %381 = load float, ptr %380, align 4, !tbaa !4
  %382 = getelementptr inbounds nuw i8, ptr %1, i64 264
  %383 = load float, ptr %382, align 4, !tbaa !4
  %384 = fpext float %383 to double
  %385 = getelementptr inbounds nuw i8, ptr %19, i64 664
  store double %384, ptr %385, align 8, !tbaa !6
  %386 = fpext float %381 to double
  %387 = getelementptr inbounds nuw i8, ptr %19, i64 672
  store double %386, ptr %387, align 8, !tbaa !6
  %388 = fpext float %379 to double
  %389 = getelementptr inbounds nuw i8, ptr %19, i64 680
  store double %388, ptr %389, align 8, !tbaa !6
  %390 = fpext float %377 to double
  %391 = getelementptr inbounds nuw i8, ptr %19, i64 688
  store double %390, ptr %391, align 8, !tbaa !6
  %392 = fpext float %375 to double
  %393 = getelementptr inbounds nuw i8, ptr %19, i64 696
  store double %392, ptr %393, align 8, !tbaa !6
  %394 = fpext float %373 to double
  %395 = getelementptr inbounds nuw i8, ptr %19, i64 704
  store double %394, ptr %395, align 8, !tbaa !6
  %396 = getelementptr inbounds nuw i8, ptr %1, i64 356
  %397 = load float, ptr %396, align 4, !tbaa !4
  %398 = getelementptr inbounds nuw i8, ptr %1, i64 352
  %399 = load float, ptr %398, align 4, !tbaa !4
  %400 = getelementptr inbounds nuw i8, ptr %1, i64 348
  %401 = load float, ptr %400, align 4, !tbaa !4
  %402 = fpext float %401 to double
  %403 = getelementptr inbounds nuw i8, ptr %19, i64 712
  store double %402, ptr %403, align 8, !tbaa !6
  %404 = fpext float %399 to double
  %405 = getelementptr inbounds nuw i8, ptr %19, i64 720
  store double %404, ptr %405, align 8, !tbaa !6
  %406 = fpext float %397 to double
  %407 = getelementptr inbounds nuw i8, ptr %19, i64 728
  store double %406, ptr %407, align 8, !tbaa !6
  %408 = getelementptr inbounds nuw i8, ptr %1, i64 320
  %409 = load float, ptr %408, align 4, !tbaa !4
  %410 = getelementptr inbounds nuw i8, ptr %1, i64 316
  %411 = load float, ptr %410, align 4, !tbaa !4
  %412 = getelementptr inbounds nuw i8, ptr %1, i64 312
  %413 = load float, ptr %412, align 4, !tbaa !4
  %414 = fpext float %413 to double
  %415 = getelementptr inbounds nuw i8, ptr %19, i64 736
  store double %414, ptr %415, align 8, !tbaa !6
  %416 = fpext float %411 to double
  %417 = getelementptr inbounds nuw i8, ptr %19, i64 744
  store double %416, ptr %417, align 8, !tbaa !6
  %418 = fpext float %409 to double
  %419 = getelementptr inbounds nuw i8, ptr %19, i64 752
  store double %418, ptr %419, align 8, !tbaa !6
  %420 = getelementptr inbounds nuw i8, ptr %1, i64 368
  %421 = load float, ptr %420, align 4, !tbaa !4
  %422 = getelementptr inbounds nuw i8, ptr %1, i64 364
  %423 = load float, ptr %422, align 4, !tbaa !4
  %424 = getelementptr inbounds nuw i8, ptr %1, i64 360
  %425 = load float, ptr %424, align 4, !tbaa !4
  %426 = fpext float %425 to double
  %427 = getelementptr inbounds nuw i8, ptr %19, i64 760
  store double %426, ptr %427, align 8, !tbaa !6
  %428 = fpext float %423 to double
  %429 = getelementptr inbounds nuw i8, ptr %19, i64 768
  store double %428, ptr %429, align 8, !tbaa !6
  %430 = fpext float %421 to double
  %431 = getelementptr inbounds nuw i8, ptr %19, i64 776
  store double %430, ptr %431, align 8, !tbaa !6
  %432 = getelementptr inbounds nuw i8, ptr %1, i64 332
  %433 = load float, ptr %432, align 4, !tbaa !4
  %434 = getelementptr inbounds nuw i8, ptr %1, i64 328
  %435 = load float, ptr %434, align 4, !tbaa !4
  %436 = getelementptr inbounds nuw i8, ptr %1, i64 324
  %437 = load float, ptr %436, align 4, !tbaa !4
  %438 = fpext float %437 to double
  %439 = getelementptr inbounds nuw i8, ptr %19, i64 784
  store double %438, ptr %439, align 8, !tbaa !6
  %440 = fpext float %435 to double
  %441 = getelementptr inbounds nuw i8, ptr %19, i64 792
  store double %440, ptr %441, align 8, !tbaa !6
  %442 = fpext float %433 to double
  %443 = getelementptr inbounds nuw i8, ptr %19, i64 800
  store double %442, ptr %443, align 8, !tbaa !6
  %444 = getelementptr inbounds nuw i8, ptr %1, i64 380
  %445 = load float, ptr %444, align 4, !tbaa !4
  %446 = getelementptr inbounds nuw i8, ptr %1, i64 376
  %447 = load float, ptr %446, align 4, !tbaa !4
  %448 = getelementptr inbounds nuw i8, ptr %1, i64 372
  %449 = load float, ptr %448, align 4, !tbaa !4
  %450 = fpext float %449 to double
  %451 = getelementptr inbounds nuw i8, ptr %19, i64 808
  store double %450, ptr %451, align 8, !tbaa !6
  %452 = fpext float %447 to double
  %453 = getelementptr inbounds nuw i8, ptr %19, i64 816
  store double %452, ptr %453, align 8, !tbaa !6
  %454 = fpext float %445 to double
  %455 = getelementptr inbounds nuw i8, ptr %19, i64 824
  store double %454, ptr %455, align 8, !tbaa !6
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
  %456 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 8)
  %457 = insertvalue { ptr, ptr, i64 } poison, ptr %456, 0
  %458 = insertvalue { ptr, ptr, i64 } %457, ptr %456, 1
  %459 = insertvalue { ptr, ptr, i64 } %458, i64 0, 2
  store ptr %456, ptr %15, align 8
  %.fca.1.gep127 = getelementptr inbounds nuw i8, ptr %15, i64 8
  store ptr %456, ptr %.fca.1.gep127, align 8
  %.fca.2.gep129 = getelementptr inbounds nuw i8, ptr %15, i64 16
  store i64 0, ptr %.fca.2.gep129, align 8
  call void @qnode_forward_0.quantum(ptr nonnull %18, ptr nonnull %17, ptr nonnull %16, ptr nonnull %15)
  ret { ptr, ptr, i64 } %459
}

define internal void @_sample_loss.cloned(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, ptr readnone captures(none) %9, ptr readonly captures(none) %10, i64 %11, ptr readnone captures(none) %12, ptr readonly captures(none) %13, i64 %14, ptr %15, ptr %16, i64 %17, i64 %18, i64 %19, ptr readnone captures(none) %20, ptr readonly captures(none) %21, i64 %22, ptr readnone captures(none) %23, ptr readonly captures(none) %24, i64 %25, ptr readnone captures(none) %26, ptr writeonly captures(none) initializes((0, 8)) %27, i64 %28) {
  %30 = load float, ptr %24, align 4, !tbaa !4
  %31 = load float, ptr %21, align 4, !tbaa !4
  %32 = load float, ptr %10, align 4, !tbaa !4
  %33 = load float, ptr %13, align 4, !tbaa !4
  %34 = tail call { ptr, ptr, i64 } @qnode_forward_0.preprocess(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, ptr %15, ptr %16, i64 %17, i64 %18, i64 %19, i64 104)
  %35 = extractvalue { ptr, ptr, i64 } %34, 1
  %36 = load double, ptr %35, align 8, !tbaa !6
  %37 = fpext float %33 to double
  %38 = fmul double %36, %37
  %39 = fpext float %32 to double
  %40 = fadd double %38, %39
  %41 = fpext float %31 to double
  %42 = fpext float %30 to double
  %.inv = fcmp ole double %40, 0.000000e+00
  %43 = select i1 %.inv, double 0.000000e+00, double %40
  %44 = fcmp uno double %40, 0.000000e+00
  %45 = tail call double @llvm.fabs.f64(double %40)
  %46 = fneg double %45
  %47 = tail call double @llvm.exp.f64(double %46)
  %48 = fadd double %47, 1.000000e+00
  %49 = tail call double @llvm.log.f64(double %48)
  %50 = fadd double %43, %49
  %51 = select i1 %44, double %40, double %50
  %52 = fmul double %40, %41
  %53 = fsub double %51, %52
  %54 = fmul double %53, %42
  %55 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %56 = ptrtoint ptr %55 to i64
  %57 = add i64 %56, 63
  %58 = and i64 %57, -64
  %59 = inttoptr i64 %58 to ptr
  store double %54, ptr %59, align 64, !tbaa !6
  store double %54, ptr %27, align 8, !tbaa !6
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

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fabs.f64(double) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.exp.f64(double) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.log.f64(double) #3

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #4

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #5

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #5

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) }
attributes #1 = { noinline }
attributes #2 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #5 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{!2, !2, i64 0}
!2 = !{!"int", !3, i64 0}
!3 = !{!"Catalyst TBAA"}
!4 = !{!5, !5, i64 0}
!5 = !{!"float", !3, i64 0}
!6 = !{!7, !7, i64 0}
!7 = !{!"double", !3, i64 0}
