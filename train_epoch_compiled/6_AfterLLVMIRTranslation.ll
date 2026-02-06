; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

@"{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}" = internal constant [54 x i8] c"{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}\00"
@LightningSimulator = internal constant [19 x i8] c"LightningSimulator\00"
@"/home/pichau/QAgents/.venv/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.so" = internal constant [107 x i8] c"/home/pichau/QAgents/.venv/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.so\00"
@enzyme_dupnoneed = linkonce constant i8 0
@enzyme_const = linkonce constant i8 0
@__enzyme_function_like_free = global [2 x ptr] [ptr @_mlir_memref_to_llvm_free, ptr @freename]
@freename = linkonce constant [5 x i8] c"free\00"
@dealloc_indices = linkonce constant [3 x i8] c"-1\00"
@__enzyme_allocation_like = global [4 x ptr] [ptr @_mlir_memref_to_llvm_alloc, ptr null, ptr @dealloc_indices, ptr @_mlir_memref_to_llvm_free]
@__enzyme_register_gradient_qnode_forward_0.quantum = global [3 x ptr] [ptr @qnode_forward_0.quantum, ptr @qnode_forward_0.quantum.augfwd, ptr @qnode_forward_0.quantum.customqgrad]
@__constant_xi64_4 = private constant i64 32, align 64
@__constant_4xi32_3 = private constant [4 x i32] [i32 13, i32 15, i32 26, i32 6], align 64
@__constant_4xi32 = private constant [4 x i32] [i32 17, i32 29, i32 16, i32 24], align 64
@__constant_xi64 = private constant i64 1, align 64
@__constant_xf32_2 = private constant float 0.000000e+00, align 64
@__constant_xf32_1 = private constant float 0x3F847AE140000000, align 64
@__constant_xf32_0 = private constant float 3.125000e-02, align 64
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

define { { ptr, ptr, i64, [3 x i64], [3 x i64] }, { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64, [1 x i64], [1 x i64] }, { ptr, ptr, i64, [1 x i64], [1 x i64] } } @jit_train_epoch_compiled(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, ptr %9, ptr %10, i64 %11, ptr %12, ptr %13, i64 %14, ptr %15, ptr %16, i64 %17, ptr %18, ptr %19, i64 %20, i64 %21, i64 %22, ptr %23, ptr %24, i64 %25, i64 %26, i64 %27, i64 %28, i64 %29, ptr %30, ptr %31, i64 %32, i64 %33, i64 %34, ptr %35, ptr %36, i64 %37, i64 %38, i64 %39) {
  %41 = insertvalue { ptr, ptr, i64 } poison, ptr %15, 0
  %42 = insertvalue { ptr, ptr, i64 } %41, ptr %16, 1
  %43 = insertvalue { ptr, ptr, i64 } %42, i64 %17, 2
  %44 = load i32, ptr %19, align 4, !tbaa !1
  %45 = getelementptr inbounds i32, ptr %19, i32 1
  %46 = load i32, ptr %45, align 4, !tbaa !1
  %47 = call ptr @_mlir_memref_to_llvm_alloc(i64 80)
  %48 = ptrtoint ptr %47 to i64
  %49 = add i64 %48, 63
  %50 = urem i64 %49, 64
  %51 = sub i64 %49, %50
  %52 = inttoptr i64 %51 to ptr
  %53 = call ptr @_mlir_memref_to_llvm_alloc(i64 80)
  %54 = ptrtoint ptr %53 to i64
  %55 = add i64 %54, 63
  %56 = urem i64 %55, 64
  %57 = sub i64 %55, %56
  %58 = inttoptr i64 %57 to ptr
  br label %59

59:                                               ; preds = %62, %40
  %60 = phi i64 [ %64, %62 ], [ 0, %40 ]
  %61 = icmp slt i64 %60, 2
  br i1 %61, label %62, label %65

62:                                               ; preds = %59
  %63 = getelementptr inbounds i64, ptr %58, i64 %60
  store i64 %60, ptr %63, align 4, !tbaa !1
  %64 = add i64 %60, 1
  br label %59

65:                                               ; preds = %59
  br label %66

66:                                               ; preds = %69, %65
  %67 = phi i64 [ %72, %69 ], [ 0, %65 ]
  %68 = icmp slt i64 %67, 2
  br i1 %68, label %69, label %73

69:                                               ; preds = %66
  %70 = load i64, ptr @__constant_xi64, align 4, !tbaa !1
  %71 = getelementptr inbounds i64, ptr %52, i64 %67
  store i64 %70, ptr %71, align 4, !tbaa !1
  %72 = add i64 %67, 1
  br label %66

73:                                               ; preds = %66
  %74 = call ptr @_mlir_memref_to_llvm_alloc(i64 80)
  %75 = ptrtoint ptr %74 to i64
  %76 = add i64 %75, 63
  %77 = urem i64 %76, 64
  %78 = sub i64 %76, %77
  %79 = inttoptr i64 %78 to ptr
  br label %80

80:                                               ; preds = %83, %73
  %81 = phi i64 [ %90, %83 ], [ 0, %73 ]
  %82 = icmp slt i64 %81, 2
  br i1 %82, label %83, label %91

83:                                               ; preds = %80
  %84 = getelementptr inbounds i64, ptr %52, i64 %81
  %85 = load i64, ptr %84, align 4, !tbaa !1
  %86 = getelementptr inbounds i64, ptr %58, i64 %81
  %87 = load i64, ptr %86, align 4, !tbaa !1
  %88 = mul i64 %85, %87
  %89 = getelementptr inbounds i64, ptr %79, i64 %81
  store i64 %88, ptr %89, align 4, !tbaa !1
  %90 = add i64 %81, 1
  br label %80

91:                                               ; preds = %80
  call void @_mlir_memref_to_llvm_free(ptr %53)
  br label %92

92:                                               ; preds = %95, %91
  %93 = phi i64 [ %98, %95 ], [ 0, %91 ]
  %94 = icmp slt i64 %93, 2
  br i1 %94, label %95, label %99

95:                                               ; preds = %92
  %96 = load i64, ptr @__constant_xi64_4, align 4, !tbaa !1
  %97 = getelementptr inbounds i64, ptr %52, i64 %93
  store i64 %96, ptr %97, align 4, !tbaa !1
  %98 = add i64 %93, 1
  br label %92

99:                                               ; preds = %92
  br label %100

100:                                              ; preds = %103, %99
  %101 = phi i64 [ %112, %103 ], [ 0, %99 ]
  %102 = icmp slt i64 %101, 2
  br i1 %102, label %103, label %113

103:                                              ; preds = %100
  %104 = getelementptr inbounds i64, ptr %79, i64 %101
  %105 = load i64, ptr %104, align 4, !tbaa !1
  %106 = getelementptr inbounds i64, ptr %52, i64 %101
  %107 = load i64, ptr %106, align 4, !tbaa !1
  %108 = lshr i64 %105, %107
  %109 = icmp ult i64 %107, 64
  %110 = select i1 %109, i64 %108, i64 0
  %111 = getelementptr inbounds i64, ptr %52, i64 %101
  store i64 %110, ptr %111, align 4, !tbaa !1
  %112 = add i64 %101, 1
  br label %100

113:                                              ; preds = %100
  %114 = call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %115 = ptrtoint ptr %114 to i64
  %116 = add i64 %115, 63
  %117 = urem i64 %116, 64
  %118 = sub i64 %116, %117
  %119 = inttoptr i64 %118 to ptr
  %120 = call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %121 = ptrtoint ptr %120 to i64
  %122 = add i64 %121, 63
  %123 = urem i64 %122, 64
  %124 = sub i64 %122, %123
  %125 = inttoptr i64 %124 to ptr
  br label %126

126:                                              ; preds = %129, %113
  %127 = phi i64 [ %134, %129 ], [ 0, %113 ]
  %128 = icmp slt i64 %127, 2
  br i1 %128, label %129, label %135

129:                                              ; preds = %126
  %130 = getelementptr inbounds i64, ptr %79, i64 %127
  %131 = load i64, ptr %130, align 4, !tbaa !1
  %132 = trunc i64 %131 to i32
  %133 = getelementptr inbounds i32, ptr %125, i64 %127
  store i32 %132, ptr %133, align 4, !tbaa !1
  %134 = add i64 %127, 1
  br label %126

135:                                              ; preds = %126
  call void @_mlir_memref_to_llvm_free(ptr %74)
  %136 = call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %137 = ptrtoint ptr %136 to i64
  %138 = add i64 %137, 63
  %139 = urem i64 %138, 64
  %140 = sub i64 %138, %139
  %141 = inttoptr i64 %140 to ptr
  br label %142

142:                                              ; preds = %145, %135
  %143 = phi i64 [ %150, %145 ], [ 0, %135 ]
  %144 = icmp slt i64 %143, 2
  br i1 %144, label %145, label %151

145:                                              ; preds = %142
  %146 = getelementptr inbounds i64, ptr %52, i64 %143
  %147 = load i64, ptr %146, align 4, !tbaa !1
  %148 = trunc i64 %147 to i32
  %149 = getelementptr inbounds i32, ptr %141, i64 %143
  store i32 %148, ptr %149, align 4, !tbaa !1
  %150 = add i64 %143, 1
  br label %142

151:                                              ; preds = %142
  call void @_mlir_memref_to_llvm_free(ptr %47)
  %152 = xor i32 %44, %46
  %153 = xor i32 %152, 466688986
  br label %154

154:                                              ; preds = %157, %151
  %155 = phi i64 [ %160, %157 ], [ 0, %151 ]
  %156 = icmp slt i64 %155, 2
  br i1 %156, label %157, label %161

157:                                              ; preds = %154
  %158 = load i32, ptr %19, align 4, !tbaa !1
  %159 = getelementptr inbounds i32, ptr %119, i64 %155
  store i32 %158, ptr %159, align 4, !tbaa !1
  %160 = add i64 %155, 1
  br label %154

161:                                              ; preds = %154
  %162 = call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %163 = ptrtoint ptr %162 to i64
  %164 = add i64 %163, 63
  %165 = urem i64 %164, 64
  %166 = sub i64 %164, %165
  %167 = inttoptr i64 %166 to ptr
  br label %168

168:                                              ; preds = %171, %161
  %169 = phi i64 [ %178, %171 ], [ 0, %161 ]
  %170 = icmp slt i64 %169, 2
  br i1 %170, label %171, label %179

171:                                              ; preds = %168
  %172 = getelementptr inbounds i32, ptr %141, i64 %169
  %173 = load i32, ptr %172, align 4, !tbaa !1
  %174 = getelementptr inbounds i32, ptr %119, i64 %169
  %175 = load i32, ptr %174, align 4, !tbaa !1
  %176 = add i32 %173, %175
  %177 = getelementptr inbounds i32, ptr %167, i64 %169
  store i32 %176, ptr %177, align 4, !tbaa !1
  %178 = add i64 %169, 1
  br label %168

179:                                              ; preds = %168
  call void @_mlir_memref_to_llvm_free(ptr %136)
  br label %180

180:                                              ; preds = %183, %179
  %181 = phi i64 [ %187, %183 ], [ 0, %179 ]
  %182 = icmp slt i64 %181, 2
  br i1 %182, label %183, label %188

183:                                              ; preds = %180
  %184 = getelementptr inbounds i32, ptr %19, i32 1
  %185 = load i32, ptr %184, align 4, !tbaa !1
  %186 = getelementptr inbounds i32, ptr %119, i64 %181
  store i32 %185, ptr %186, align 4, !tbaa !1
  %187 = add i64 %181, 1
  br label %180

188:                                              ; preds = %180
  br label %189

189:                                              ; preds = %192, %188
  %190 = phi i64 [ %199, %192 ], [ 0, %188 ]
  %191 = icmp slt i64 %190, 2
  br i1 %191, label %192, label %200

192:                                              ; preds = %189
  %193 = getelementptr inbounds i32, ptr %125, i64 %190
  %194 = load i32, ptr %193, align 4, !tbaa !1
  %195 = getelementptr inbounds i32, ptr %119, i64 %190
  %196 = load i32, ptr %195, align 4, !tbaa !1
  %197 = add i32 %194, %196
  %198 = getelementptr inbounds i32, ptr %119, i64 %190
  store i32 %197, ptr %198, align 4, !tbaa !1
  %199 = add i64 %190, 1
  br label %189

200:                                              ; preds = %189
  call void @_mlir_memref_to_llvm_free(ptr %120)
  %201 = getelementptr inbounds i32, ptr %19, i32 1
  %202 = load i32, ptr %201, align 4, !tbaa !1
  %203 = load i32, ptr %19, align 4, !tbaa !1
  %204 = call ptr @_mlir_memref_to_llvm_alloc(i64 80)
  %205 = ptrtoint ptr %204 to i64
  %206 = add i64 %205, 63
  %207 = urem i64 %206, 64
  %208 = sub i64 %206, %207
  %209 = inttoptr i64 %208 to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr %209, ptr @__constant_4xi32_3, i64 16, i1 false)
  %210 = call ptr @_mlir_memref_to_llvm_alloc(i64 80)
  %211 = ptrtoint ptr %210 to i64
  %212 = add i64 %211, 63
  %213 = urem i64 %212, 64
  %214 = sub i64 %212, %213
  %215 = inttoptr i64 %214 to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr %215, ptr @__constant_4xi32, i64 16, i1 false)
  %216 = call ptr @_mlir_memref_to_llvm_alloc(i64 68)
  %217 = ptrtoint ptr %216 to i64
  %218 = add i64 %217, 63
  %219 = urem i64 %218, 64
  %220 = sub i64 %218, %219
  %221 = inttoptr i64 %220 to ptr
  %222 = call ptr @_mlir_memref_to_llvm_alloc(i64 68)
  %223 = ptrtoint ptr %222 to i64
  %224 = add i64 %223, 63
  %225 = urem i64 %224, 64
  %226 = sub i64 %224, %225
  %227 = inttoptr i64 %226 to ptr
  %228 = call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %229 = ptrtoint ptr %228 to i64
  %230 = add i64 %229, 63
  %231 = urem i64 %230, 64
  %232 = sub i64 %230, %231
  %233 = inttoptr i64 %232 to ptr
  %234 = call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %235 = ptrtoint ptr %234 to i64
  %236 = add i64 %235, 63
  %237 = urem i64 %236, 64
  %238 = sub i64 %236, %237
  %239 = inttoptr i64 %238 to ptr
  %240 = call ptr @_mlir_memref_to_llvm_alloc(i64 68)
  %241 = ptrtoint ptr %240 to i64
  %242 = add i64 %241, 63
  %243 = urem i64 %242, 64
  %244 = sub i64 %242, %243
  %245 = inttoptr i64 %244 to ptr
  %246 = call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %247 = ptrtoint ptr %246 to i64
  %248 = add i64 %247, 63
  %249 = urem i64 %248, 64
  %250 = sub i64 %248, %249
  %251 = inttoptr i64 %250 to ptr
  %252 = call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %253 = ptrtoint ptr %252 to i64
  %254 = add i64 %253, 63
  %255 = urem i64 %254, 64
  %256 = sub i64 %254, %255
  %257 = inttoptr i64 %256 to ptr
  %258 = call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %259 = ptrtoint ptr %258 to i64
  %260 = add i64 %259, 63
  %261 = urem i64 %260, 64
  %262 = sub i64 %260, %261
  %263 = inttoptr i64 %262 to ptr
  %264 = call ptr @_mlir_memref_to_llvm_alloc(i64 68)
  %265 = ptrtoint ptr %264 to i64
  %266 = add i64 %265, 63
  %267 = urem i64 %266, 64
  %268 = sub i64 %266, %267
  %269 = inttoptr i64 %268 to ptr
  %270 = call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %271 = ptrtoint ptr %270 to i64
  %272 = add i64 %271, 63
  %273 = urem i64 %272, 64
  %274 = sub i64 %272, %273
  %275 = inttoptr i64 %274 to ptr
  %276 = call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %277 = ptrtoint ptr %276 to i64
  %278 = add i64 %277, 63
  %279 = urem i64 %278, 64
  %280 = sub i64 %278, %279
  %281 = inttoptr i64 %280 to ptr
  %282 = call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %283 = ptrtoint ptr %282 to i64
  %284 = add i64 %283, 63
  %285 = urem i64 %284, 64
  %286 = sub i64 %284, %285
  %287 = inttoptr i64 %286 to ptr
  %288 = call ptr @_mlir_memref_to_llvm_alloc(i64 68)
  %289 = ptrtoint ptr %288 to i64
  %290 = add i64 %289, 63
  %291 = urem i64 %290, 64
  %292 = sub i64 %290, %291
  %293 = inttoptr i64 %292 to ptr
  %294 = call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %295 = ptrtoint ptr %294 to i64
  %296 = add i64 %295, 63
  %297 = urem i64 %296, 64
  %298 = sub i64 %296, %297
  %299 = inttoptr i64 %298 to ptr
  %300 = call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %301 = ptrtoint ptr %300 to i64
  %302 = add i64 %301, 63
  %303 = urem i64 %302, 64
  %304 = sub i64 %302, %303
  %305 = inttoptr i64 %304 to ptr
  %306 = call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %307 = ptrtoint ptr %306 to i64
  %308 = add i64 %307, 63
  %309 = urem i64 %308, 64
  %310 = sub i64 %308, %309
  %311 = inttoptr i64 %310 to ptr
  %312 = call ptr @_mlir_memref_to_llvm_alloc(i64 68)
  %313 = ptrtoint ptr %312 to i64
  %314 = add i64 %313, 63
  %315 = urem i64 %314, 64
  %316 = sub i64 %314, %315
  %317 = inttoptr i64 %316 to ptr
  %318 = call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %319 = ptrtoint ptr %318 to i64
  %320 = add i64 %319, 63
  %321 = urem i64 %320, 64
  %322 = sub i64 %320, %321
  %323 = inttoptr i64 %322 to ptr
  %324 = call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %325 = ptrtoint ptr %324 to i64
  %326 = add i64 %325, 63
  %327 = urem i64 %326, 64
  %328 = sub i64 %326, %327
  %329 = inttoptr i64 %328 to ptr
  %330 = call ptr @_mlir_memref_to_llvm_alloc(i64 68)
  %331 = ptrtoint ptr %330 to i64
  %332 = add i64 %331, 63
  %333 = urem i64 %332, 64
  %334 = sub i64 %332, %333
  %335 = inttoptr i64 %334 to ptr
  %336 = call ptr @_mlir_memref_to_llvm_alloc(i64 8)
  %337 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %336, 0
  %338 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %337, ptr %336, 1
  %339 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %338, i64 0, 2
  %340 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %339, i64 2, 3, 0
  %341 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %340, i64 1, 4, 0
  call void @llvm.memcpy.p0.p0.i64(ptr %336, ptr %119, i64 8, i1 false)
  call void @_mlir_memref_to_llvm_free(ptr %114)
  %342 = call ptr @_mlir_memref_to_llvm_alloc(i64 8)
  %343 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %342, 0
  %344 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %343, ptr %342, 1
  %345 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %344, i64 0, 2
  %346 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %345, i64 2, 3, 0
  %347 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %346, i64 1, 4, 0
  call void @llvm.memcpy.p0.p0.i64(ptr %342, ptr %167, i64 8, i1 false)
  call void @_mlir_memref_to_llvm_free(ptr %162)
  %348 = call ptr @_mlir_memref_to_llvm_alloc(i64 16)
  %349 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %348, 0
  %350 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %349, ptr %348, 1
  %351 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %350, i64 0, 2
  %352 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %351, i64 4, 3, 0
  %353 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %352, i64 1, 4, 0
  call void @llvm.memcpy.p0.p0.i64(ptr %348, ptr %209, i64 16, i1 false)
  call void @_mlir_memref_to_llvm_free(ptr %204)
  %354 = call ptr @_mlir_memref_to_llvm_alloc(i64 16)
  %355 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %354, 0
  %356 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %355, ptr %354, 1
  %357 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %356, i64 0, 2
  %358 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %357, i64 4, 3, 0
  %359 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %358, i64 1, 4, 0
  call void @llvm.memcpy.p0.p0.i64(ptr %354, ptr %215, i64 16, i1 false)
  call void @_mlir_memref_to_llvm_free(ptr %210)
  br label %360

360:                                              ; preds = %789, %200
  %361 = phi i64 [ %840, %789 ], [ 0, %200 ]
  %362 = phi i64 [ %372, %789 ], [ 0, %200 ]
  %363 = phi { ptr, ptr, i64, [1 x i64], [1 x i64] } [ %827, %789 ], [ %347, %200 ]
  %364 = phi { ptr, ptr, i64, [1 x i64], [1 x i64] } [ %821, %789 ], [ %341, %200 ]
  %365 = phi i32 [ %366, %789 ], [ %202, %200 ]
  %366 = phi i32 [ %367, %789 ], [ %153, %200 ]
  %367 = phi i32 [ %365, %789 ], [ %203, %200 ]
  %368 = phi { ptr, ptr, i64, [1 x i64], [1 x i64] } [ %833, %789 ], [ %353, %200 ]
  %369 = phi { ptr, ptr, i64, [1 x i64], [1 x i64] } [ %839, %789 ], [ %359, %200 ]
  %370 = icmp slt i64 %361, 5
  br i1 %370, label %371, label %841

371:                                              ; preds = %360
  store i32 %365, ptr %221, align 4, !tbaa !1
  store i32 %366, ptr %227, align 4, !tbaa !1
  %372 = add i64 %362, 1
  %373 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %368, 1
  %374 = load i32, ptr %373, align 4, !tbaa !1
  %375 = call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %376 = ptrtoint ptr %375 to i64
  %377 = add i64 %376, 63
  %378 = urem i64 %377, 64
  %379 = sub i64 %377, %378
  %380 = inttoptr i64 %379 to ptr
  br label %381

381:                                              ; preds = %384, %371
  %382 = phi i64 [ %393, %384 ], [ 0, %371 ]
  %383 = icmp slt i64 %382, 2
  br i1 %383, label %384, label %394

384:                                              ; preds = %381
  %385 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %363, 1
  %386 = getelementptr inbounds i32, ptr %385, i64 %382
  %387 = load i32, ptr %386, align 4, !tbaa !1
  %388 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %364, 1
  %389 = getelementptr inbounds i32, ptr %388, i64 %382
  %390 = load i32, ptr %389, align 4, !tbaa !1
  %391 = add i32 %387, %390
  %392 = getelementptr inbounds i32, ptr %233, i64 %382
  store i32 %391, ptr %392, align 4, !tbaa !1
  %393 = add i64 %382, 1
  br label %381

394:                                              ; preds = %381
  %395 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %363, 0
  call void @_mlir_memref_to_llvm_free(ptr %395)
  br label %396

396:                                              ; preds = %399, %394
  %397 = phi i64 [ %402, %399 ], [ 0, %394 ]
  %398 = icmp slt i64 %397, 2
  br i1 %398, label %399, label %403

399:                                              ; preds = %396
  %400 = load i32, ptr %373, align 4, !tbaa !1
  %401 = getelementptr inbounds i32, ptr %380, i64 %397
  store i32 %400, ptr %401, align 4, !tbaa !1
  %402 = add i64 %397, 1
  br label %396

403:                                              ; preds = %396
  br label %404

404:                                              ; preds = %407, %403
  %405 = phi i64 [ %417, %407 ], [ 0, %403 ]
  %406 = icmp slt i64 %405, 2
  br i1 %406, label %407, label %418

407:                                              ; preds = %404
  %408 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %364, 1
  %409 = getelementptr inbounds i32, ptr %408, i64 %405
  %410 = load i32, ptr %409, align 4, !tbaa !1
  %411 = getelementptr inbounds i32, ptr %380, i64 %405
  %412 = load i32, ptr %411, align 4, !tbaa !1
  %413 = shl i32 %410, %412
  %414 = icmp ult i32 %412, 32
  %415 = select i1 %414, i32 %413, i32 0
  %416 = getelementptr inbounds i32, ptr %239, i64 %405
  store i32 %415, ptr %416, align 4, !tbaa !1
  %417 = add i64 %405, 1
  br label %404

418:                                              ; preds = %404
  %419 = sub i32 32, %374
  store i32 %419, ptr %245, align 4, !tbaa !1
  br label %420

420:                                              ; preds = %423, %418
  %421 = phi i64 [ %426, %423 ], [ 0, %418 ]
  %422 = icmp slt i64 %421, 2
  br i1 %422, label %423, label %427

423:                                              ; preds = %420
  %424 = load i32, ptr %245, align 4, !tbaa !1
  %425 = getelementptr inbounds i32, ptr %380, i64 %421
  store i32 %424, ptr %425, align 4, !tbaa !1
  %426 = add i64 %421, 1
  br label %420

427:                                              ; preds = %420
  br label %428

428:                                              ; preds = %431, %427
  %429 = phi i64 [ %441, %431 ], [ 0, %427 ]
  %430 = icmp slt i64 %429, 2
  br i1 %430, label %431, label %442

431:                                              ; preds = %428
  %432 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %364, 1
  %433 = getelementptr inbounds i32, ptr %432, i64 %429
  %434 = load i32, ptr %433, align 4, !tbaa !1
  %435 = getelementptr inbounds i32, ptr %380, i64 %429
  %436 = load i32, ptr %435, align 4, !tbaa !1
  %437 = lshr i32 %434, %436
  %438 = icmp ult i32 %436, 32
  %439 = select i1 %438, i32 %437, i32 0
  %440 = getelementptr inbounds i32, ptr %380, i64 %429
  store i32 %439, ptr %440, align 4, !tbaa !1
  %441 = add i64 %429, 1
  br label %428

442:                                              ; preds = %428
  %443 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %364, 0
  call void @_mlir_memref_to_llvm_free(ptr %443)
  br label %444

444:                                              ; preds = %447, %442
  %445 = phi i64 [ %454, %447 ], [ 0, %442 ]
  %446 = icmp slt i64 %445, 2
  br i1 %446, label %447, label %455

447:                                              ; preds = %444
  %448 = getelementptr inbounds i32, ptr %239, i64 %445
  %449 = load i32, ptr %448, align 4, !tbaa !1
  %450 = getelementptr inbounds i32, ptr %380, i64 %445
  %451 = load i32, ptr %450, align 4, !tbaa !1
  %452 = or i32 %449, %451
  %453 = getelementptr inbounds i32, ptr %380, i64 %445
  store i32 %452, ptr %453, align 4, !tbaa !1
  %454 = add i64 %445, 1
  br label %444

455:                                              ; preds = %444
  br label %456

456:                                              ; preds = %459, %455
  %457 = phi i64 [ %466, %459 ], [ 0, %455 ]
  %458 = icmp slt i64 %457, 2
  br i1 %458, label %459, label %467

459:                                              ; preds = %456
  %460 = getelementptr inbounds i32, ptr %233, i64 %457
  %461 = load i32, ptr %460, align 4, !tbaa !1
  %462 = getelementptr inbounds i32, ptr %380, i64 %457
  %463 = load i32, ptr %462, align 4, !tbaa !1
  %464 = xor i32 %461, %463
  %465 = getelementptr inbounds i32, ptr %251, i64 %457
  store i32 %464, ptr %465, align 4, !tbaa !1
  %466 = add i64 %457, 1
  br label %456

467:                                              ; preds = %456
  %468 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %368, 1
  %469 = getelementptr inbounds i32, ptr %468, i32 1
  %470 = load i32, ptr %469, align 4, !tbaa !1
  br label %471

471:                                              ; preds = %474, %467
  %472 = phi i64 [ %481, %474 ], [ 0, %467 ]
  %473 = icmp slt i64 %472, 2
  br i1 %473, label %474, label %482

474:                                              ; preds = %471
  %475 = getelementptr inbounds i32, ptr %233, i64 %472
  %476 = load i32, ptr %475, align 4, !tbaa !1
  %477 = getelementptr inbounds i32, ptr %251, i64 %472
  %478 = load i32, ptr %477, align 4, !tbaa !1
  %479 = add i32 %476, %478
  %480 = getelementptr inbounds i32, ptr %257, i64 %472
  store i32 %479, ptr %480, align 4, !tbaa !1
  %481 = add i64 %472, 1
  br label %471

482:                                              ; preds = %471
  br label %483

483:                                              ; preds = %486, %482
  %484 = phi i64 [ %490, %486 ], [ 0, %482 ]
  %485 = icmp slt i64 %484, 2
  br i1 %485, label %486, label %491

486:                                              ; preds = %483
  %487 = getelementptr inbounds i32, ptr %468, i32 1
  %488 = load i32, ptr %487, align 4, !tbaa !1
  %489 = getelementptr inbounds i32, ptr %380, i64 %484
  store i32 %488, ptr %489, align 4, !tbaa !1
  %490 = add i64 %484, 1
  br label %483

491:                                              ; preds = %483
  br label %492

492:                                              ; preds = %495, %491
  %493 = phi i64 [ %504, %495 ], [ 0, %491 ]
  %494 = icmp slt i64 %493, 2
  br i1 %494, label %495, label %505

495:                                              ; preds = %492
  %496 = getelementptr inbounds i32, ptr %251, i64 %493
  %497 = load i32, ptr %496, align 4, !tbaa !1
  %498 = getelementptr inbounds i32, ptr %380, i64 %493
  %499 = load i32, ptr %498, align 4, !tbaa !1
  %500 = shl i32 %497, %499
  %501 = icmp ult i32 %499, 32
  %502 = select i1 %501, i32 %500, i32 0
  %503 = getelementptr inbounds i32, ptr %263, i64 %493
  store i32 %502, ptr %503, align 4, !tbaa !1
  %504 = add i64 %493, 1
  br label %492

505:                                              ; preds = %492
  %506 = sub i32 32, %470
  store i32 %506, ptr %269, align 4, !tbaa !1
  br label %507

507:                                              ; preds = %510, %505
  %508 = phi i64 [ %513, %510 ], [ 0, %505 ]
  %509 = icmp slt i64 %508, 2
  br i1 %509, label %510, label %514

510:                                              ; preds = %507
  %511 = load i32, ptr %269, align 4, !tbaa !1
  %512 = getelementptr inbounds i32, ptr %380, i64 %508
  store i32 %511, ptr %512, align 4, !tbaa !1
  %513 = add i64 %508, 1
  br label %507

514:                                              ; preds = %507
  br label %515

515:                                              ; preds = %518, %514
  %516 = phi i64 [ %527, %518 ], [ 0, %514 ]
  %517 = icmp slt i64 %516, 2
  br i1 %517, label %518, label %528

518:                                              ; preds = %515
  %519 = getelementptr inbounds i32, ptr %251, i64 %516
  %520 = load i32, ptr %519, align 4, !tbaa !1
  %521 = getelementptr inbounds i32, ptr %380, i64 %516
  %522 = load i32, ptr %521, align 4, !tbaa !1
  %523 = lshr i32 %520, %522
  %524 = icmp ult i32 %522, 32
  %525 = select i1 %524, i32 %523, i32 0
  %526 = getelementptr inbounds i32, ptr %380, i64 %516
  store i32 %525, ptr %526, align 4, !tbaa !1
  %527 = add i64 %516, 1
  br label %515

528:                                              ; preds = %515
  br label %529

529:                                              ; preds = %532, %528
  %530 = phi i64 [ %539, %532 ], [ 0, %528 ]
  %531 = icmp slt i64 %530, 2
  br i1 %531, label %532, label %540

532:                                              ; preds = %529
  %533 = getelementptr inbounds i32, ptr %263, i64 %530
  %534 = load i32, ptr %533, align 4, !tbaa !1
  %535 = getelementptr inbounds i32, ptr %380, i64 %530
  %536 = load i32, ptr %535, align 4, !tbaa !1
  %537 = or i32 %534, %536
  %538 = getelementptr inbounds i32, ptr %380, i64 %530
  store i32 %537, ptr %538, align 4, !tbaa !1
  %539 = add i64 %530, 1
  br label %529

540:                                              ; preds = %529
  br label %541

541:                                              ; preds = %544, %540
  %542 = phi i64 [ %551, %544 ], [ 0, %540 ]
  %543 = icmp slt i64 %542, 2
  br i1 %543, label %544, label %552

544:                                              ; preds = %541
  %545 = getelementptr inbounds i32, ptr %257, i64 %542
  %546 = load i32, ptr %545, align 4, !tbaa !1
  %547 = getelementptr inbounds i32, ptr %380, i64 %542
  %548 = load i32, ptr %547, align 4, !tbaa !1
  %549 = xor i32 %546, %548
  %550 = getelementptr inbounds i32, ptr %275, i64 %542
  store i32 %549, ptr %550, align 4, !tbaa !1
  %551 = add i64 %542, 1
  br label %541

552:                                              ; preds = %541
  %553 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %368, 1
  %554 = getelementptr inbounds i32, ptr %553, i32 2
  %555 = load i32, ptr %554, align 4, !tbaa !1
  br label %556

556:                                              ; preds = %559, %552
  %557 = phi i64 [ %566, %559 ], [ 0, %552 ]
  %558 = icmp slt i64 %557, 2
  br i1 %558, label %559, label %567

559:                                              ; preds = %556
  %560 = getelementptr inbounds i32, ptr %257, i64 %557
  %561 = load i32, ptr %560, align 4, !tbaa !1
  %562 = getelementptr inbounds i32, ptr %275, i64 %557
  %563 = load i32, ptr %562, align 4, !tbaa !1
  %564 = add i32 %561, %563
  %565 = getelementptr inbounds i32, ptr %281, i64 %557
  store i32 %564, ptr %565, align 4, !tbaa !1
  %566 = add i64 %557, 1
  br label %556

567:                                              ; preds = %556
  br label %568

568:                                              ; preds = %571, %567
  %569 = phi i64 [ %575, %571 ], [ 0, %567 ]
  %570 = icmp slt i64 %569, 2
  br i1 %570, label %571, label %576

571:                                              ; preds = %568
  %572 = getelementptr inbounds i32, ptr %553, i32 2
  %573 = load i32, ptr %572, align 4, !tbaa !1
  %574 = getelementptr inbounds i32, ptr %380, i64 %569
  store i32 %573, ptr %574, align 4, !tbaa !1
  %575 = add i64 %569, 1
  br label %568

576:                                              ; preds = %568
  br label %577

577:                                              ; preds = %580, %576
  %578 = phi i64 [ %589, %580 ], [ 0, %576 ]
  %579 = icmp slt i64 %578, 2
  br i1 %579, label %580, label %590

580:                                              ; preds = %577
  %581 = getelementptr inbounds i32, ptr %275, i64 %578
  %582 = load i32, ptr %581, align 4, !tbaa !1
  %583 = getelementptr inbounds i32, ptr %380, i64 %578
  %584 = load i32, ptr %583, align 4, !tbaa !1
  %585 = shl i32 %582, %584
  %586 = icmp ult i32 %584, 32
  %587 = select i1 %586, i32 %585, i32 0
  %588 = getelementptr inbounds i32, ptr %287, i64 %578
  store i32 %587, ptr %588, align 4, !tbaa !1
  %589 = add i64 %578, 1
  br label %577

590:                                              ; preds = %577
  %591 = sub i32 32, %555
  store i32 %591, ptr %293, align 4, !tbaa !1
  br label %592

592:                                              ; preds = %595, %590
  %593 = phi i64 [ %598, %595 ], [ 0, %590 ]
  %594 = icmp slt i64 %593, 2
  br i1 %594, label %595, label %599

595:                                              ; preds = %592
  %596 = load i32, ptr %293, align 4, !tbaa !1
  %597 = getelementptr inbounds i32, ptr %380, i64 %593
  store i32 %596, ptr %597, align 4, !tbaa !1
  %598 = add i64 %593, 1
  br label %592

599:                                              ; preds = %592
  br label %600

600:                                              ; preds = %603, %599
  %601 = phi i64 [ %612, %603 ], [ 0, %599 ]
  %602 = icmp slt i64 %601, 2
  br i1 %602, label %603, label %613

603:                                              ; preds = %600
  %604 = getelementptr inbounds i32, ptr %275, i64 %601
  %605 = load i32, ptr %604, align 4, !tbaa !1
  %606 = getelementptr inbounds i32, ptr %380, i64 %601
  %607 = load i32, ptr %606, align 4, !tbaa !1
  %608 = lshr i32 %605, %607
  %609 = icmp ult i32 %607, 32
  %610 = select i1 %609, i32 %608, i32 0
  %611 = getelementptr inbounds i32, ptr %380, i64 %601
  store i32 %610, ptr %611, align 4, !tbaa !1
  %612 = add i64 %601, 1
  br label %600

613:                                              ; preds = %600
  br label %614

614:                                              ; preds = %617, %613
  %615 = phi i64 [ %624, %617 ], [ 0, %613 ]
  %616 = icmp slt i64 %615, 2
  br i1 %616, label %617, label %625

617:                                              ; preds = %614
  %618 = getelementptr inbounds i32, ptr %287, i64 %615
  %619 = load i32, ptr %618, align 4, !tbaa !1
  %620 = getelementptr inbounds i32, ptr %380, i64 %615
  %621 = load i32, ptr %620, align 4, !tbaa !1
  %622 = or i32 %619, %621
  %623 = getelementptr inbounds i32, ptr %380, i64 %615
  store i32 %622, ptr %623, align 4, !tbaa !1
  %624 = add i64 %615, 1
  br label %614

625:                                              ; preds = %614
  br label %626

626:                                              ; preds = %629, %625
  %627 = phi i64 [ %636, %629 ], [ 0, %625 ]
  %628 = icmp slt i64 %627, 2
  br i1 %628, label %629, label %637

629:                                              ; preds = %626
  %630 = getelementptr inbounds i32, ptr %281, i64 %627
  %631 = load i32, ptr %630, align 4, !tbaa !1
  %632 = getelementptr inbounds i32, ptr %380, i64 %627
  %633 = load i32, ptr %632, align 4, !tbaa !1
  %634 = xor i32 %631, %633
  %635 = getelementptr inbounds i32, ptr %299, i64 %627
  store i32 %634, ptr %635, align 4, !tbaa !1
  %636 = add i64 %627, 1
  br label %626

637:                                              ; preds = %626
  %638 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %368, 1
  %639 = getelementptr inbounds i32, ptr %638, i32 3
  %640 = load i32, ptr %639, align 4, !tbaa !1
  br label %641

641:                                              ; preds = %644, %637
  %642 = phi i64 [ %651, %644 ], [ 0, %637 ]
  %643 = icmp slt i64 %642, 2
  br i1 %643, label %644, label %652

644:                                              ; preds = %641
  %645 = getelementptr inbounds i32, ptr %281, i64 %642
  %646 = load i32, ptr %645, align 4, !tbaa !1
  %647 = getelementptr inbounds i32, ptr %299, i64 %642
  %648 = load i32, ptr %647, align 4, !tbaa !1
  %649 = add i32 %646, %648
  %650 = getelementptr inbounds i32, ptr %305, i64 %642
  store i32 %649, ptr %650, align 4, !tbaa !1
  %651 = add i64 %642, 1
  br label %641

652:                                              ; preds = %641
  br label %653

653:                                              ; preds = %656, %652
  %654 = phi i64 [ %660, %656 ], [ 0, %652 ]
  %655 = icmp slt i64 %654, 2
  br i1 %655, label %656, label %661

656:                                              ; preds = %653
  %657 = getelementptr inbounds i32, ptr %638, i32 3
  %658 = load i32, ptr %657, align 4, !tbaa !1
  %659 = getelementptr inbounds i32, ptr %380, i64 %654
  store i32 %658, ptr %659, align 4, !tbaa !1
  %660 = add i64 %654, 1
  br label %653

661:                                              ; preds = %653
  br label %662

662:                                              ; preds = %665, %661
  %663 = phi i64 [ %674, %665 ], [ 0, %661 ]
  %664 = icmp slt i64 %663, 2
  br i1 %664, label %665, label %675

665:                                              ; preds = %662
  %666 = getelementptr inbounds i32, ptr %299, i64 %663
  %667 = load i32, ptr %666, align 4, !tbaa !1
  %668 = getelementptr inbounds i32, ptr %380, i64 %663
  %669 = load i32, ptr %668, align 4, !tbaa !1
  %670 = shl i32 %667, %669
  %671 = icmp ult i32 %669, 32
  %672 = select i1 %671, i32 %670, i32 0
  %673 = getelementptr inbounds i32, ptr %311, i64 %663
  store i32 %672, ptr %673, align 4, !tbaa !1
  %674 = add i64 %663, 1
  br label %662

675:                                              ; preds = %662
  %676 = sub i32 32, %640
  store i32 %676, ptr %317, align 4, !tbaa !1
  br label %677

677:                                              ; preds = %680, %675
  %678 = phi i64 [ %683, %680 ], [ 0, %675 ]
  %679 = icmp slt i64 %678, 2
  br i1 %679, label %680, label %684

680:                                              ; preds = %677
  %681 = load i32, ptr %317, align 4, !tbaa !1
  %682 = getelementptr inbounds i32, ptr %380, i64 %678
  store i32 %681, ptr %682, align 4, !tbaa !1
  %683 = add i64 %678, 1
  br label %677

684:                                              ; preds = %677
  br label %685

685:                                              ; preds = %688, %684
  %686 = phi i64 [ %697, %688 ], [ 0, %684 ]
  %687 = icmp slt i64 %686, 2
  br i1 %687, label %688, label %698

688:                                              ; preds = %685
  %689 = getelementptr inbounds i32, ptr %299, i64 %686
  %690 = load i32, ptr %689, align 4, !tbaa !1
  %691 = getelementptr inbounds i32, ptr %380, i64 %686
  %692 = load i32, ptr %691, align 4, !tbaa !1
  %693 = lshr i32 %690, %692
  %694 = icmp ult i32 %692, 32
  %695 = select i1 %694, i32 %693, i32 0
  %696 = getelementptr inbounds i32, ptr %380, i64 %686
  store i32 %695, ptr %696, align 4, !tbaa !1
  %697 = add i64 %686, 1
  br label %685

698:                                              ; preds = %685
  br label %699

699:                                              ; preds = %702, %698
  %700 = phi i64 [ %709, %702 ], [ 0, %698 ]
  %701 = icmp slt i64 %700, 2
  br i1 %701, label %702, label %710

702:                                              ; preds = %699
  %703 = getelementptr inbounds i32, ptr %311, i64 %700
  %704 = load i32, ptr %703, align 4, !tbaa !1
  %705 = getelementptr inbounds i32, ptr %380, i64 %700
  %706 = load i32, ptr %705, align 4, !tbaa !1
  %707 = or i32 %704, %706
  %708 = getelementptr inbounds i32, ptr %380, i64 %700
  store i32 %707, ptr %708, align 4, !tbaa !1
  %709 = add i64 %700, 1
  br label %699

710:                                              ; preds = %699
  br label %711

711:                                              ; preds = %714, %710
  %712 = phi i64 [ %721, %714 ], [ 0, %710 ]
  %713 = icmp slt i64 %712, 2
  br i1 %713, label %714, label %722

714:                                              ; preds = %711
  %715 = getelementptr inbounds i32, ptr %305, i64 %712
  %716 = load i32, ptr %715, align 4, !tbaa !1
  %717 = getelementptr inbounds i32, ptr %380, i64 %712
  %718 = load i32, ptr %717, align 4, !tbaa !1
  %719 = xor i32 %716, %718
  %720 = getelementptr inbounds i32, ptr %323, i64 %712
  store i32 %719, ptr %720, align 4, !tbaa !1
  %721 = add i64 %712, 1
  br label %711

722:                                              ; preds = %711
  br label %723

723:                                              ; preds = %726, %722
  %724 = phi i64 [ %729, %726 ], [ 0, %722 ]
  %725 = icmp slt i64 %724, 2
  br i1 %725, label %726, label %730

726:                                              ; preds = %723
  %727 = load i32, ptr %221, align 4, !tbaa !1
  %728 = getelementptr inbounds i32, ptr %380, i64 %724
  store i32 %727, ptr %728, align 4, !tbaa !1
  %729 = add i64 %724, 1
  br label %723

730:                                              ; preds = %723
  %731 = call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %732 = ptrtoint ptr %731 to i64
  %733 = add i64 %732, 63
  %734 = urem i64 %733, 64
  %735 = sub i64 %733, %734
  %736 = inttoptr i64 %735 to ptr
  br label %737

737:                                              ; preds = %740, %730
  %738 = phi i64 [ %747, %740 ], [ 0, %730 ]
  %739 = icmp slt i64 %738, 2
  br i1 %739, label %740, label %748

740:                                              ; preds = %737
  %741 = getelementptr inbounds i32, ptr %305, i64 %738
  %742 = load i32, ptr %741, align 4, !tbaa !1
  %743 = getelementptr inbounds i32, ptr %380, i64 %738
  %744 = load i32, ptr %743, align 4, !tbaa !1
  %745 = add i32 %742, %744
  %746 = getelementptr inbounds i32, ptr %736, i64 %738
  store i32 %745, ptr %746, align 4, !tbaa !1
  %747 = add i64 %738, 1
  br label %737

748:                                              ; preds = %737
  br label %749

749:                                              ; preds = %752, %748
  %750 = phi i64 [ %755, %752 ], [ 0, %748 ]
  %751 = icmp slt i64 %750, 2
  br i1 %751, label %752, label %756

752:                                              ; preds = %749
  %753 = load i32, ptr %227, align 4, !tbaa !1
  %754 = getelementptr inbounds i32, ptr %380, i64 %750
  store i32 %753, ptr %754, align 4, !tbaa !1
  %755 = add i64 %750, 1
  br label %749

756:                                              ; preds = %749
  br label %757

757:                                              ; preds = %760, %756
  %758 = phi i64 [ %767, %760 ], [ 0, %756 ]
  %759 = icmp slt i64 %758, 2
  br i1 %759, label %760, label %768

760:                                              ; preds = %757
  %761 = getelementptr inbounds i32, ptr %323, i64 %758
  %762 = load i32, ptr %761, align 4, !tbaa !1
  %763 = getelementptr inbounds i32, ptr %380, i64 %758
  %764 = load i32, ptr %763, align 4, !tbaa !1
  %765 = add i32 %762, %764
  %766 = getelementptr inbounds i32, ptr %329, i64 %758
  store i32 %765, ptr %766, align 4, !tbaa !1
  %767 = add i64 %758, 1
  br label %757

768:                                              ; preds = %757
  %769 = trunc i64 %372 to i32
  store i32 %769, ptr %335, align 4, !tbaa !1
  br label %770

770:                                              ; preds = %773, %768
  %771 = phi i64 [ %776, %773 ], [ 0, %768 ]
  %772 = icmp slt i64 %771, 2
  br i1 %772, label %773, label %777

773:                                              ; preds = %770
  %774 = load i32, ptr %335, align 4, !tbaa !1
  %775 = getelementptr inbounds i32, ptr %380, i64 %771
  store i32 %774, ptr %775, align 4, !tbaa !1
  %776 = add i64 %771, 1
  br label %770

777:                                              ; preds = %770
  br label %778

778:                                              ; preds = %781, %777
  %779 = phi i64 [ %788, %781 ], [ 0, %777 ]
  %780 = icmp slt i64 %779, 2
  br i1 %780, label %781, label %789

781:                                              ; preds = %778
  %782 = getelementptr inbounds i32, ptr %329, i64 %779
  %783 = load i32, ptr %782, align 4, !tbaa !1
  %784 = getelementptr inbounds i32, ptr %380, i64 %779
  %785 = load i32, ptr %784, align 4, !tbaa !1
  %786 = add i32 %783, %785
  %787 = getelementptr inbounds i32, ptr %380, i64 %779
  store i32 %786, ptr %787, align 4, !tbaa !1
  %788 = add i64 %779, 1
  br label %778

789:                                              ; preds = %778
  %790 = call ptr @_mlir_memref_to_llvm_alloc(i64 80)
  %791 = ptrtoint ptr %790 to i64
  %792 = add i64 %791, 63
  %793 = urem i64 %792, 64
  %794 = sub i64 %792, %793
  %795 = inttoptr i64 %794 to ptr
  %796 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %369, 3, 0
  %797 = mul i64 %796, 1
  %798 = mul i64 %797, 4
  %799 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %369, 1
  %800 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %369, 2
  %801 = getelementptr inbounds i32, ptr %799, i64 %800
  call void @llvm.memcpy.p0.p0.i64(ptr %795, ptr %801, i64 %798, i1 false)
  %802 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %369, 0
  call void @_mlir_memref_to_llvm_free(ptr %802)
  %803 = call ptr @_mlir_memref_to_llvm_alloc(i64 80)
  %804 = ptrtoint ptr %803 to i64
  %805 = add i64 %804, 63
  %806 = urem i64 %805, 64
  %807 = sub i64 %805, %806
  %808 = inttoptr i64 %807 to ptr
  %809 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %368, 3, 0
  %810 = mul i64 %809, 1
  %811 = mul i64 %810, 4
  %812 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %368, 1
  %813 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %368, 2
  %814 = getelementptr inbounds i32, ptr %812, i64 %813
  call void @llvm.memcpy.p0.p0.i64(ptr %808, ptr %814, i64 %811, i1 false)
  %815 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %368, 0
  call void @_mlir_memref_to_llvm_free(ptr %815)
  %816 = call ptr @_mlir_memref_to_llvm_alloc(i64 8)
  %817 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %816, 0
  %818 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %817, ptr %816, 1
  %819 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %818, i64 0, 2
  %820 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %819, i64 2, 3, 0
  %821 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %820, i64 1, 4, 0
  call void @llvm.memcpy.p0.p0.i64(ptr %816, ptr %380, i64 8, i1 false)
  call void @_mlir_memref_to_llvm_free(ptr %375)
  %822 = call ptr @_mlir_memref_to_llvm_alloc(i64 8)
  %823 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %822, 0
  %824 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %823, ptr %822, 1
  %825 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %824, i64 0, 2
  %826 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %825, i64 2, 3, 0
  %827 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %826, i64 1, 4, 0
  call void @llvm.memcpy.p0.p0.i64(ptr %822, ptr %736, i64 8, i1 false)
  call void @_mlir_memref_to_llvm_free(ptr %731)
  %828 = call ptr @_mlir_memref_to_llvm_alloc(i64 16)
  %829 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %828, 0
  %830 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %829, ptr %828, 1
  %831 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %830, i64 0, 2
  %832 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %831, i64 4, 3, 0
  %833 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %832, i64 1, 4, 0
  call void @llvm.memcpy.p0.p0.i64(ptr %828, ptr %795, i64 16, i1 false)
  call void @_mlir_memref_to_llvm_free(ptr %790)
  %834 = call ptr @_mlir_memref_to_llvm_alloc(i64 16)
  %835 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %834, 0
  %836 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %835, ptr %834, 1
  %837 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %836, i64 0, 2
  %838 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %837, i64 4, 3, 0
  %839 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %838, i64 1, 4, 0
  call void @llvm.memcpy.p0.p0.i64(ptr %834, ptr %808, i64 16, i1 false)
  call void @_mlir_memref_to_llvm_free(ptr %803)
  %840 = add i64 %361, 1
  br label %360

841:                                              ; preds = %360
  %842 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %369, 0
  call void @_mlir_memref_to_llvm_free(ptr %842)
  %843 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %368, 0
  call void @_mlir_memref_to_llvm_free(ptr %843)
  call void @_mlir_memref_to_llvm_free(ptr %330)
  call void @_mlir_memref_to_llvm_free(ptr %324)
  call void @_mlir_memref_to_llvm_free(ptr %318)
  call void @_mlir_memref_to_llvm_free(ptr %312)
  call void @_mlir_memref_to_llvm_free(ptr %306)
  call void @_mlir_memref_to_llvm_free(ptr %300)
  call void @_mlir_memref_to_llvm_free(ptr %294)
  call void @_mlir_memref_to_llvm_free(ptr %288)
  call void @_mlir_memref_to_llvm_free(ptr %282)
  call void @_mlir_memref_to_llvm_free(ptr %276)
  call void @_mlir_memref_to_llvm_free(ptr %270)
  call void @_mlir_memref_to_llvm_free(ptr %264)
  call void @_mlir_memref_to_llvm_free(ptr %258)
  call void @_mlir_memref_to_llvm_free(ptr %252)
  call void @_mlir_memref_to_llvm_free(ptr %246)
  call void @_mlir_memref_to_llvm_free(ptr %240)
  call void @_mlir_memref_to_llvm_free(ptr %234)
  call void @_mlir_memref_to_llvm_free(ptr %228)
  call void @_mlir_memref_to_llvm_free(ptr %222)
  call void @_mlir_memref_to_llvm_free(ptr %216)
  %844 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %363, 1
  %845 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %364, 1
  %846 = call ptr @_mlir_memref_to_llvm_alloc(i64 80)
  %847 = ptrtoint ptr %846 to i64
  %848 = add i64 %847, 63
  %849 = urem i64 %848, 64
  %850 = sub i64 %848, %849
  %851 = inttoptr i64 %850 to ptr
  br label %852

852:                                              ; preds = %877, %841
  %853 = phi i64 [ %878, %877 ], [ 0, %841 ]
  %854 = icmp slt i64 %853, 2
  br i1 %854, label %855, label %879

855:                                              ; preds = %852
  br label %856

856:                                              ; preds = %872, %855
  %857 = phi i64 [ %876, %872 ], [ 0, %855 ]
  %858 = icmp slt i64 %857, 2
  br i1 %858, label %859, label %877

859:                                              ; preds = %856
  %860 = icmp ult i64 %857, 1
  br i1 %860, label %861, label %865

861:                                              ; preds = %859
  %862 = add i64 %853, %857
  %863 = getelementptr inbounds i32, ptr %844, i64 %862
  %864 = load i32, ptr %863, align 4, !tbaa !1
  br label %870

865:                                              ; preds = %859
  %866 = sub i64 %857, 1
  %867 = add i64 %853, %866
  %868 = getelementptr inbounds i32, ptr %845, i64 %867
  %869 = load i32, ptr %868, align 4, !tbaa !1
  br label %870

870:                                              ; preds = %861, %865
  %871 = phi i32 [ %869, %865 ], [ %864, %861 ]
  br label %872

872:                                              ; preds = %870
  %873 = mul i64 %853, 2
  %874 = add i64 %873, %857
  %875 = getelementptr inbounds i32, ptr %851, i64 %874
  store i32 %871, ptr %875, align 4, !tbaa !1
  %876 = add i64 %857, 1
  br label %856

877:                                              ; preds = %856
  %878 = add i64 %853, 1
  br label %852

879:                                              ; preds = %852
  %880 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %363, 0
  call void @_mlir_memref_to_llvm_free(ptr %880)
  %881 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %364, 0
  call void @_mlir_memref_to_llvm_free(ptr %881)
  %882 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %846, 0
  %883 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %882, ptr %851, 1
  %884 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %883, i64 0, 2
  %885 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %884, i64 2, 3, 0
  %886 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %885, i64 1, 4, 0
  %887 = getelementptr inbounds i32, ptr %851, i32 2
  %888 = load i32, ptr %887, align 4, !tbaa !1
  %889 = getelementptr inbounds i32, ptr %851, i32 3
  %890 = load i32, ptr %889, align 4, !tbaa !1
  %891 = call ptr @_mlir_memref_to_llvm_alloc(i64 80)
  %892 = ptrtoint ptr %891 to i64
  %893 = add i64 %892, 63
  %894 = urem i64 %893, 64
  %895 = sub i64 %893, %894
  %896 = inttoptr i64 %895 to ptr
  %897 = call ptr @_mlir_memref_to_llvm_alloc(i64 80)
  %898 = ptrtoint ptr %897 to i64
  %899 = add i64 %898, 63
  %900 = urem i64 %899, 64
  %901 = sub i64 %899, %900
  %902 = inttoptr i64 %901 to ptr
  br label %903

903:                                              ; preds = %906, %879
  %904 = phi i64 [ %908, %906 ], [ 0, %879 ]
  %905 = icmp slt i64 %904, 2
  br i1 %905, label %906, label %909

906:                                              ; preds = %903
  %907 = getelementptr inbounds i64, ptr %902, i64 %904
  store i64 %904, ptr %907, align 4, !tbaa !1
  %908 = add i64 %904, 1
  br label %903

909:                                              ; preds = %903
  br label %910

910:                                              ; preds = %913, %909
  %911 = phi i64 [ %916, %913 ], [ 0, %909 ]
  %912 = icmp slt i64 %911, 2
  br i1 %912, label %913, label %917

913:                                              ; preds = %910
  %914 = load i64, ptr @__constant_xi64, align 4, !tbaa !1
  %915 = getelementptr inbounds i64, ptr %896, i64 %911
  store i64 %914, ptr %915, align 4, !tbaa !1
  %916 = add i64 %911, 1
  br label %910

917:                                              ; preds = %910
  %918 = call ptr @_mlir_memref_to_llvm_alloc(i64 80)
  %919 = ptrtoint ptr %918 to i64
  %920 = add i64 %919, 63
  %921 = urem i64 %920, 64
  %922 = sub i64 %920, %921
  %923 = inttoptr i64 %922 to ptr
  br label %924

924:                                              ; preds = %927, %917
  %925 = phi i64 [ %934, %927 ], [ 0, %917 ]
  %926 = icmp slt i64 %925, 2
  br i1 %926, label %927, label %935

927:                                              ; preds = %924
  %928 = getelementptr inbounds i64, ptr %896, i64 %925
  %929 = load i64, ptr %928, align 4, !tbaa !1
  %930 = getelementptr inbounds i64, ptr %902, i64 %925
  %931 = load i64, ptr %930, align 4, !tbaa !1
  %932 = mul i64 %929, %931
  %933 = getelementptr inbounds i64, ptr %923, i64 %925
  store i64 %932, ptr %933, align 4, !tbaa !1
  %934 = add i64 %925, 1
  br label %924

935:                                              ; preds = %924
  call void @_mlir_memref_to_llvm_free(ptr %897)
  br label %936

936:                                              ; preds = %939, %935
  %937 = phi i64 [ %942, %939 ], [ 0, %935 ]
  %938 = icmp slt i64 %937, 2
  br i1 %938, label %939, label %943

939:                                              ; preds = %936
  %940 = load i64, ptr @__constant_xi64_4, align 4, !tbaa !1
  %941 = getelementptr inbounds i64, ptr %896, i64 %937
  store i64 %940, ptr %941, align 4, !tbaa !1
  %942 = add i64 %937, 1
  br label %936

943:                                              ; preds = %936
  br label %944

944:                                              ; preds = %947, %943
  %945 = phi i64 [ %956, %947 ], [ 0, %943 ]
  %946 = icmp slt i64 %945, 2
  br i1 %946, label %947, label %957

947:                                              ; preds = %944
  %948 = getelementptr inbounds i64, ptr %923, i64 %945
  %949 = load i64, ptr %948, align 4, !tbaa !1
  %950 = getelementptr inbounds i64, ptr %896, i64 %945
  %951 = load i64, ptr %950, align 4, !tbaa !1
  %952 = lshr i64 %949, %951
  %953 = icmp ult i64 %951, 64
  %954 = select i1 %953, i64 %952, i64 0
  %955 = getelementptr inbounds i64, ptr %896, i64 %945
  store i64 %954, ptr %955, align 4, !tbaa !1
  %956 = add i64 %945, 1
  br label %944

957:                                              ; preds = %944
  %958 = call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %959 = ptrtoint ptr %958 to i64
  %960 = add i64 %959, 63
  %961 = urem i64 %960, 64
  %962 = sub i64 %960, %961
  %963 = inttoptr i64 %962 to ptr
  %964 = call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %965 = ptrtoint ptr %964 to i64
  %966 = add i64 %965, 63
  %967 = urem i64 %966, 64
  %968 = sub i64 %966, %967
  %969 = inttoptr i64 %968 to ptr
  br label %970

970:                                              ; preds = %973, %957
  %971 = phi i64 [ %978, %973 ], [ 0, %957 ]
  %972 = icmp slt i64 %971, 2
  br i1 %972, label %973, label %979

973:                                              ; preds = %970
  %974 = getelementptr inbounds i64, ptr %923, i64 %971
  %975 = load i64, ptr %974, align 4, !tbaa !1
  %976 = trunc i64 %975 to i32
  %977 = getelementptr inbounds i32, ptr %969, i64 %971
  store i32 %976, ptr %977, align 4, !tbaa !1
  %978 = add i64 %971, 1
  br label %970

979:                                              ; preds = %970
  call void @_mlir_memref_to_llvm_free(ptr %918)
  %980 = call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %981 = ptrtoint ptr %980 to i64
  %982 = add i64 %981, 63
  %983 = urem i64 %982, 64
  %984 = sub i64 %982, %983
  %985 = inttoptr i64 %984 to ptr
  br label %986

986:                                              ; preds = %989, %979
  %987 = phi i64 [ %994, %989 ], [ 0, %979 ]
  %988 = icmp slt i64 %987, 2
  br i1 %988, label %989, label %995

989:                                              ; preds = %986
  %990 = getelementptr inbounds i64, ptr %896, i64 %987
  %991 = load i64, ptr %990, align 4, !tbaa !1
  %992 = trunc i64 %991 to i32
  %993 = getelementptr inbounds i32, ptr %985, i64 %987
  store i32 %992, ptr %993, align 4, !tbaa !1
  %994 = add i64 %987, 1
  br label %986

995:                                              ; preds = %986
  call void @_mlir_memref_to_llvm_free(ptr %891)
  %996 = xor i32 %888, %890
  %997 = xor i32 %996, 466688986
  br label %998

998:                                              ; preds = %1001, %995
  %999 = phi i64 [ %1005, %1001 ], [ 0, %995 ]
  %1000 = icmp slt i64 %999, 2
  br i1 %1000, label %1001, label %1006

1001:                                             ; preds = %998
  %1002 = getelementptr inbounds i32, ptr %851, i32 2
  %1003 = load i32, ptr %1002, align 4, !tbaa !1
  %1004 = getelementptr inbounds i32, ptr %963, i64 %999
  store i32 %1003, ptr %1004, align 4, !tbaa !1
  %1005 = add i64 %999, 1
  br label %998

1006:                                             ; preds = %998
  %1007 = call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %1008 = ptrtoint ptr %1007 to i64
  %1009 = add i64 %1008, 63
  %1010 = urem i64 %1009, 64
  %1011 = sub i64 %1009, %1010
  %1012 = inttoptr i64 %1011 to ptr
  br label %1013

1013:                                             ; preds = %1016, %1006
  %1014 = phi i64 [ %1023, %1016 ], [ 0, %1006 ]
  %1015 = icmp slt i64 %1014, 2
  br i1 %1015, label %1016, label %1024

1016:                                             ; preds = %1013
  %1017 = getelementptr inbounds i32, ptr %985, i64 %1014
  %1018 = load i32, ptr %1017, align 4, !tbaa !1
  %1019 = getelementptr inbounds i32, ptr %963, i64 %1014
  %1020 = load i32, ptr %1019, align 4, !tbaa !1
  %1021 = add i32 %1018, %1020
  %1022 = getelementptr inbounds i32, ptr %1012, i64 %1014
  store i32 %1021, ptr %1022, align 4, !tbaa !1
  %1023 = add i64 %1014, 1
  br label %1013

1024:                                             ; preds = %1013
  call void @_mlir_memref_to_llvm_free(ptr %980)
  br label %1025

1025:                                             ; preds = %1028, %1024
  %1026 = phi i64 [ %1032, %1028 ], [ 0, %1024 ]
  %1027 = icmp slt i64 %1026, 2
  br i1 %1027, label %1028, label %1033

1028:                                             ; preds = %1025
  %1029 = getelementptr inbounds i32, ptr %851, i32 3
  %1030 = load i32, ptr %1029, align 4, !tbaa !1
  %1031 = getelementptr inbounds i32, ptr %963, i64 %1026
  store i32 %1030, ptr %1031, align 4, !tbaa !1
  %1032 = add i64 %1026, 1
  br label %1025

1033:                                             ; preds = %1025
  br label %1034

1034:                                             ; preds = %1037, %1033
  %1035 = phi i64 [ %1044, %1037 ], [ 0, %1033 ]
  %1036 = icmp slt i64 %1035, 2
  br i1 %1036, label %1037, label %1045

1037:                                             ; preds = %1034
  %1038 = getelementptr inbounds i32, ptr %969, i64 %1035
  %1039 = load i32, ptr %1038, align 4, !tbaa !1
  %1040 = getelementptr inbounds i32, ptr %963, i64 %1035
  %1041 = load i32, ptr %1040, align 4, !tbaa !1
  %1042 = add i32 %1039, %1041
  %1043 = getelementptr inbounds i32, ptr %963, i64 %1035
  store i32 %1042, ptr %1043, align 4, !tbaa !1
  %1044 = add i64 %1035, 1
  br label %1034

1045:                                             ; preds = %1034
  call void @_mlir_memref_to_llvm_free(ptr %964)
  %1046 = getelementptr inbounds i32, ptr %851, i32 3
  %1047 = load i32, ptr %1046, align 4, !tbaa !1
  %1048 = getelementptr inbounds i32, ptr %851, i32 2
  %1049 = load i32, ptr %1048, align 4, !tbaa !1
  %1050 = call ptr @_mlir_memref_to_llvm_alloc(i64 80)
  %1051 = ptrtoint ptr %1050 to i64
  %1052 = add i64 %1051, 63
  %1053 = urem i64 %1052, 64
  %1054 = sub i64 %1052, %1053
  %1055 = inttoptr i64 %1054 to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr %1055, ptr @__constant_4xi32_3, i64 16, i1 false)
  %1056 = call ptr @_mlir_memref_to_llvm_alloc(i64 80)
  %1057 = ptrtoint ptr %1056 to i64
  %1058 = add i64 %1057, 63
  %1059 = urem i64 %1058, 64
  %1060 = sub i64 %1058, %1059
  %1061 = inttoptr i64 %1060 to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr %1061, ptr @__constant_4xi32, i64 16, i1 false)
  %1062 = call ptr @_mlir_memref_to_llvm_alloc(i64 68)
  %1063 = ptrtoint ptr %1062 to i64
  %1064 = add i64 %1063, 63
  %1065 = urem i64 %1064, 64
  %1066 = sub i64 %1064, %1065
  %1067 = inttoptr i64 %1066 to ptr
  %1068 = call ptr @_mlir_memref_to_llvm_alloc(i64 68)
  %1069 = ptrtoint ptr %1068 to i64
  %1070 = add i64 %1069, 63
  %1071 = urem i64 %1070, 64
  %1072 = sub i64 %1070, %1071
  %1073 = inttoptr i64 %1072 to ptr
  %1074 = call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %1075 = ptrtoint ptr %1074 to i64
  %1076 = add i64 %1075, 63
  %1077 = urem i64 %1076, 64
  %1078 = sub i64 %1076, %1077
  %1079 = inttoptr i64 %1078 to ptr
  %1080 = call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %1081 = ptrtoint ptr %1080 to i64
  %1082 = add i64 %1081, 63
  %1083 = urem i64 %1082, 64
  %1084 = sub i64 %1082, %1083
  %1085 = inttoptr i64 %1084 to ptr
  %1086 = call ptr @_mlir_memref_to_llvm_alloc(i64 68)
  %1087 = ptrtoint ptr %1086 to i64
  %1088 = add i64 %1087, 63
  %1089 = urem i64 %1088, 64
  %1090 = sub i64 %1088, %1089
  %1091 = inttoptr i64 %1090 to ptr
  %1092 = call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %1093 = ptrtoint ptr %1092 to i64
  %1094 = add i64 %1093, 63
  %1095 = urem i64 %1094, 64
  %1096 = sub i64 %1094, %1095
  %1097 = inttoptr i64 %1096 to ptr
  %1098 = call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %1099 = ptrtoint ptr %1098 to i64
  %1100 = add i64 %1099, 63
  %1101 = urem i64 %1100, 64
  %1102 = sub i64 %1100, %1101
  %1103 = inttoptr i64 %1102 to ptr
  %1104 = call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %1105 = ptrtoint ptr %1104 to i64
  %1106 = add i64 %1105, 63
  %1107 = urem i64 %1106, 64
  %1108 = sub i64 %1106, %1107
  %1109 = inttoptr i64 %1108 to ptr
  %1110 = call ptr @_mlir_memref_to_llvm_alloc(i64 68)
  %1111 = ptrtoint ptr %1110 to i64
  %1112 = add i64 %1111, 63
  %1113 = urem i64 %1112, 64
  %1114 = sub i64 %1112, %1113
  %1115 = inttoptr i64 %1114 to ptr
  %1116 = call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %1117 = ptrtoint ptr %1116 to i64
  %1118 = add i64 %1117, 63
  %1119 = urem i64 %1118, 64
  %1120 = sub i64 %1118, %1119
  %1121 = inttoptr i64 %1120 to ptr
  %1122 = call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %1123 = ptrtoint ptr %1122 to i64
  %1124 = add i64 %1123, 63
  %1125 = urem i64 %1124, 64
  %1126 = sub i64 %1124, %1125
  %1127 = inttoptr i64 %1126 to ptr
  %1128 = call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %1129 = ptrtoint ptr %1128 to i64
  %1130 = add i64 %1129, 63
  %1131 = urem i64 %1130, 64
  %1132 = sub i64 %1130, %1131
  %1133 = inttoptr i64 %1132 to ptr
  %1134 = call ptr @_mlir_memref_to_llvm_alloc(i64 68)
  %1135 = ptrtoint ptr %1134 to i64
  %1136 = add i64 %1135, 63
  %1137 = urem i64 %1136, 64
  %1138 = sub i64 %1136, %1137
  %1139 = inttoptr i64 %1138 to ptr
  %1140 = call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %1141 = ptrtoint ptr %1140 to i64
  %1142 = add i64 %1141, 63
  %1143 = urem i64 %1142, 64
  %1144 = sub i64 %1142, %1143
  %1145 = inttoptr i64 %1144 to ptr
  %1146 = call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %1147 = ptrtoint ptr %1146 to i64
  %1148 = add i64 %1147, 63
  %1149 = urem i64 %1148, 64
  %1150 = sub i64 %1148, %1149
  %1151 = inttoptr i64 %1150 to ptr
  %1152 = call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %1153 = ptrtoint ptr %1152 to i64
  %1154 = add i64 %1153, 63
  %1155 = urem i64 %1154, 64
  %1156 = sub i64 %1154, %1155
  %1157 = inttoptr i64 %1156 to ptr
  %1158 = call ptr @_mlir_memref_to_llvm_alloc(i64 68)
  %1159 = ptrtoint ptr %1158 to i64
  %1160 = add i64 %1159, 63
  %1161 = urem i64 %1160, 64
  %1162 = sub i64 %1160, %1161
  %1163 = inttoptr i64 %1162 to ptr
  %1164 = call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %1165 = ptrtoint ptr %1164 to i64
  %1166 = add i64 %1165, 63
  %1167 = urem i64 %1166, 64
  %1168 = sub i64 %1166, %1167
  %1169 = inttoptr i64 %1168 to ptr
  %1170 = call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %1171 = ptrtoint ptr %1170 to i64
  %1172 = add i64 %1171, 63
  %1173 = urem i64 %1172, 64
  %1174 = sub i64 %1172, %1173
  %1175 = inttoptr i64 %1174 to ptr
  %1176 = call ptr @_mlir_memref_to_llvm_alloc(i64 68)
  %1177 = ptrtoint ptr %1176 to i64
  %1178 = add i64 %1177, 63
  %1179 = urem i64 %1178, 64
  %1180 = sub i64 %1178, %1179
  %1181 = inttoptr i64 %1180 to ptr
  %1182 = call ptr @_mlir_memref_to_llvm_alloc(i64 8)
  %1183 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %1182, 0
  %1184 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1183, ptr %1182, 1
  %1185 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1184, i64 0, 2
  %1186 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1185, i64 2, 3, 0
  %1187 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1186, i64 1, 4, 0
  call void @llvm.memcpy.p0.p0.i64(ptr %1182, ptr %963, i64 8, i1 false)
  call void @_mlir_memref_to_llvm_free(ptr %958)
  %1188 = call ptr @_mlir_memref_to_llvm_alloc(i64 8)
  %1189 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %1188, 0
  %1190 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1189, ptr %1188, 1
  %1191 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1190, i64 0, 2
  %1192 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1191, i64 2, 3, 0
  %1193 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1192, i64 1, 4, 0
  call void @llvm.memcpy.p0.p0.i64(ptr %1188, ptr %1012, i64 8, i1 false)
  call void @_mlir_memref_to_llvm_free(ptr %1007)
  %1194 = call ptr @_mlir_memref_to_llvm_alloc(i64 16)
  %1195 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %1194, 0
  %1196 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1195, ptr %1194, 1
  %1197 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1196, i64 0, 2
  %1198 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1197, i64 4, 3, 0
  %1199 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1198, i64 1, 4, 0
  call void @llvm.memcpy.p0.p0.i64(ptr %1194, ptr %1055, i64 16, i1 false)
  call void @_mlir_memref_to_llvm_free(ptr %1050)
  %1200 = call ptr @_mlir_memref_to_llvm_alloc(i64 16)
  %1201 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %1200, 0
  %1202 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1201, ptr %1200, 1
  %1203 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1202, i64 0, 2
  %1204 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1203, i64 4, 3, 0
  %1205 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1204, i64 1, 4, 0
  call void @llvm.memcpy.p0.p0.i64(ptr %1200, ptr %1061, i64 16, i1 false)
  call void @_mlir_memref_to_llvm_free(ptr %1056)
  br label %1206

1206:                                             ; preds = %1635, %1045
  %1207 = phi i64 [ %1686, %1635 ], [ 0, %1045 ]
  %1208 = phi i64 [ %1218, %1635 ], [ 0, %1045 ]
  %1209 = phi { ptr, ptr, i64, [1 x i64], [1 x i64] } [ %1673, %1635 ], [ %1193, %1045 ]
  %1210 = phi { ptr, ptr, i64, [1 x i64], [1 x i64] } [ %1667, %1635 ], [ %1187, %1045 ]
  %1211 = phi i32 [ %1212, %1635 ], [ %1047, %1045 ]
  %1212 = phi i32 [ %1213, %1635 ], [ %997, %1045 ]
  %1213 = phi i32 [ %1211, %1635 ], [ %1049, %1045 ]
  %1214 = phi { ptr, ptr, i64, [1 x i64], [1 x i64] } [ %1679, %1635 ], [ %1199, %1045 ]
  %1215 = phi { ptr, ptr, i64, [1 x i64], [1 x i64] } [ %1685, %1635 ], [ %1205, %1045 ]
  %1216 = icmp slt i64 %1207, 5
  br i1 %1216, label %1217, label %1687

1217:                                             ; preds = %1206
  store i32 %1211, ptr %1067, align 4, !tbaa !1
  store i32 %1212, ptr %1073, align 4, !tbaa !1
  %1218 = add i64 %1208, 1
  %1219 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1214, 1
  %1220 = load i32, ptr %1219, align 4, !tbaa !1
  %1221 = call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %1222 = ptrtoint ptr %1221 to i64
  %1223 = add i64 %1222, 63
  %1224 = urem i64 %1223, 64
  %1225 = sub i64 %1223, %1224
  %1226 = inttoptr i64 %1225 to ptr
  br label %1227

1227:                                             ; preds = %1230, %1217
  %1228 = phi i64 [ %1239, %1230 ], [ 0, %1217 ]
  %1229 = icmp slt i64 %1228, 2
  br i1 %1229, label %1230, label %1240

1230:                                             ; preds = %1227
  %1231 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1209, 1
  %1232 = getelementptr inbounds i32, ptr %1231, i64 %1228
  %1233 = load i32, ptr %1232, align 4, !tbaa !1
  %1234 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1210, 1
  %1235 = getelementptr inbounds i32, ptr %1234, i64 %1228
  %1236 = load i32, ptr %1235, align 4, !tbaa !1
  %1237 = add i32 %1233, %1236
  %1238 = getelementptr inbounds i32, ptr %1079, i64 %1228
  store i32 %1237, ptr %1238, align 4, !tbaa !1
  %1239 = add i64 %1228, 1
  br label %1227

1240:                                             ; preds = %1227
  %1241 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1209, 0
  call void @_mlir_memref_to_llvm_free(ptr %1241)
  br label %1242

1242:                                             ; preds = %1245, %1240
  %1243 = phi i64 [ %1248, %1245 ], [ 0, %1240 ]
  %1244 = icmp slt i64 %1243, 2
  br i1 %1244, label %1245, label %1249

1245:                                             ; preds = %1242
  %1246 = load i32, ptr %1219, align 4, !tbaa !1
  %1247 = getelementptr inbounds i32, ptr %1226, i64 %1243
  store i32 %1246, ptr %1247, align 4, !tbaa !1
  %1248 = add i64 %1243, 1
  br label %1242

1249:                                             ; preds = %1242
  br label %1250

1250:                                             ; preds = %1253, %1249
  %1251 = phi i64 [ %1263, %1253 ], [ 0, %1249 ]
  %1252 = icmp slt i64 %1251, 2
  br i1 %1252, label %1253, label %1264

1253:                                             ; preds = %1250
  %1254 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1210, 1
  %1255 = getelementptr inbounds i32, ptr %1254, i64 %1251
  %1256 = load i32, ptr %1255, align 4, !tbaa !1
  %1257 = getelementptr inbounds i32, ptr %1226, i64 %1251
  %1258 = load i32, ptr %1257, align 4, !tbaa !1
  %1259 = shl i32 %1256, %1258
  %1260 = icmp ult i32 %1258, 32
  %1261 = select i1 %1260, i32 %1259, i32 0
  %1262 = getelementptr inbounds i32, ptr %1085, i64 %1251
  store i32 %1261, ptr %1262, align 4, !tbaa !1
  %1263 = add i64 %1251, 1
  br label %1250

1264:                                             ; preds = %1250
  %1265 = sub i32 32, %1220
  store i32 %1265, ptr %1091, align 4, !tbaa !1
  br label %1266

1266:                                             ; preds = %1269, %1264
  %1267 = phi i64 [ %1272, %1269 ], [ 0, %1264 ]
  %1268 = icmp slt i64 %1267, 2
  br i1 %1268, label %1269, label %1273

1269:                                             ; preds = %1266
  %1270 = load i32, ptr %1091, align 4, !tbaa !1
  %1271 = getelementptr inbounds i32, ptr %1226, i64 %1267
  store i32 %1270, ptr %1271, align 4, !tbaa !1
  %1272 = add i64 %1267, 1
  br label %1266

1273:                                             ; preds = %1266
  br label %1274

1274:                                             ; preds = %1277, %1273
  %1275 = phi i64 [ %1287, %1277 ], [ 0, %1273 ]
  %1276 = icmp slt i64 %1275, 2
  br i1 %1276, label %1277, label %1288

1277:                                             ; preds = %1274
  %1278 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1210, 1
  %1279 = getelementptr inbounds i32, ptr %1278, i64 %1275
  %1280 = load i32, ptr %1279, align 4, !tbaa !1
  %1281 = getelementptr inbounds i32, ptr %1226, i64 %1275
  %1282 = load i32, ptr %1281, align 4, !tbaa !1
  %1283 = lshr i32 %1280, %1282
  %1284 = icmp ult i32 %1282, 32
  %1285 = select i1 %1284, i32 %1283, i32 0
  %1286 = getelementptr inbounds i32, ptr %1226, i64 %1275
  store i32 %1285, ptr %1286, align 4, !tbaa !1
  %1287 = add i64 %1275, 1
  br label %1274

1288:                                             ; preds = %1274
  %1289 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1210, 0
  call void @_mlir_memref_to_llvm_free(ptr %1289)
  br label %1290

1290:                                             ; preds = %1293, %1288
  %1291 = phi i64 [ %1300, %1293 ], [ 0, %1288 ]
  %1292 = icmp slt i64 %1291, 2
  br i1 %1292, label %1293, label %1301

1293:                                             ; preds = %1290
  %1294 = getelementptr inbounds i32, ptr %1085, i64 %1291
  %1295 = load i32, ptr %1294, align 4, !tbaa !1
  %1296 = getelementptr inbounds i32, ptr %1226, i64 %1291
  %1297 = load i32, ptr %1296, align 4, !tbaa !1
  %1298 = or i32 %1295, %1297
  %1299 = getelementptr inbounds i32, ptr %1226, i64 %1291
  store i32 %1298, ptr %1299, align 4, !tbaa !1
  %1300 = add i64 %1291, 1
  br label %1290

1301:                                             ; preds = %1290
  br label %1302

1302:                                             ; preds = %1305, %1301
  %1303 = phi i64 [ %1312, %1305 ], [ 0, %1301 ]
  %1304 = icmp slt i64 %1303, 2
  br i1 %1304, label %1305, label %1313

1305:                                             ; preds = %1302
  %1306 = getelementptr inbounds i32, ptr %1079, i64 %1303
  %1307 = load i32, ptr %1306, align 4, !tbaa !1
  %1308 = getelementptr inbounds i32, ptr %1226, i64 %1303
  %1309 = load i32, ptr %1308, align 4, !tbaa !1
  %1310 = xor i32 %1307, %1309
  %1311 = getelementptr inbounds i32, ptr %1097, i64 %1303
  store i32 %1310, ptr %1311, align 4, !tbaa !1
  %1312 = add i64 %1303, 1
  br label %1302

1313:                                             ; preds = %1302
  %1314 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1214, 1
  %1315 = getelementptr inbounds i32, ptr %1314, i32 1
  %1316 = load i32, ptr %1315, align 4, !tbaa !1
  br label %1317

1317:                                             ; preds = %1320, %1313
  %1318 = phi i64 [ %1327, %1320 ], [ 0, %1313 ]
  %1319 = icmp slt i64 %1318, 2
  br i1 %1319, label %1320, label %1328

1320:                                             ; preds = %1317
  %1321 = getelementptr inbounds i32, ptr %1079, i64 %1318
  %1322 = load i32, ptr %1321, align 4, !tbaa !1
  %1323 = getelementptr inbounds i32, ptr %1097, i64 %1318
  %1324 = load i32, ptr %1323, align 4, !tbaa !1
  %1325 = add i32 %1322, %1324
  %1326 = getelementptr inbounds i32, ptr %1103, i64 %1318
  store i32 %1325, ptr %1326, align 4, !tbaa !1
  %1327 = add i64 %1318, 1
  br label %1317

1328:                                             ; preds = %1317
  br label %1329

1329:                                             ; preds = %1332, %1328
  %1330 = phi i64 [ %1336, %1332 ], [ 0, %1328 ]
  %1331 = icmp slt i64 %1330, 2
  br i1 %1331, label %1332, label %1337

1332:                                             ; preds = %1329
  %1333 = getelementptr inbounds i32, ptr %1314, i32 1
  %1334 = load i32, ptr %1333, align 4, !tbaa !1
  %1335 = getelementptr inbounds i32, ptr %1226, i64 %1330
  store i32 %1334, ptr %1335, align 4, !tbaa !1
  %1336 = add i64 %1330, 1
  br label %1329

1337:                                             ; preds = %1329
  br label %1338

1338:                                             ; preds = %1341, %1337
  %1339 = phi i64 [ %1350, %1341 ], [ 0, %1337 ]
  %1340 = icmp slt i64 %1339, 2
  br i1 %1340, label %1341, label %1351

1341:                                             ; preds = %1338
  %1342 = getelementptr inbounds i32, ptr %1097, i64 %1339
  %1343 = load i32, ptr %1342, align 4, !tbaa !1
  %1344 = getelementptr inbounds i32, ptr %1226, i64 %1339
  %1345 = load i32, ptr %1344, align 4, !tbaa !1
  %1346 = shl i32 %1343, %1345
  %1347 = icmp ult i32 %1345, 32
  %1348 = select i1 %1347, i32 %1346, i32 0
  %1349 = getelementptr inbounds i32, ptr %1109, i64 %1339
  store i32 %1348, ptr %1349, align 4, !tbaa !1
  %1350 = add i64 %1339, 1
  br label %1338

1351:                                             ; preds = %1338
  %1352 = sub i32 32, %1316
  store i32 %1352, ptr %1115, align 4, !tbaa !1
  br label %1353

1353:                                             ; preds = %1356, %1351
  %1354 = phi i64 [ %1359, %1356 ], [ 0, %1351 ]
  %1355 = icmp slt i64 %1354, 2
  br i1 %1355, label %1356, label %1360

1356:                                             ; preds = %1353
  %1357 = load i32, ptr %1115, align 4, !tbaa !1
  %1358 = getelementptr inbounds i32, ptr %1226, i64 %1354
  store i32 %1357, ptr %1358, align 4, !tbaa !1
  %1359 = add i64 %1354, 1
  br label %1353

1360:                                             ; preds = %1353
  br label %1361

1361:                                             ; preds = %1364, %1360
  %1362 = phi i64 [ %1373, %1364 ], [ 0, %1360 ]
  %1363 = icmp slt i64 %1362, 2
  br i1 %1363, label %1364, label %1374

1364:                                             ; preds = %1361
  %1365 = getelementptr inbounds i32, ptr %1097, i64 %1362
  %1366 = load i32, ptr %1365, align 4, !tbaa !1
  %1367 = getelementptr inbounds i32, ptr %1226, i64 %1362
  %1368 = load i32, ptr %1367, align 4, !tbaa !1
  %1369 = lshr i32 %1366, %1368
  %1370 = icmp ult i32 %1368, 32
  %1371 = select i1 %1370, i32 %1369, i32 0
  %1372 = getelementptr inbounds i32, ptr %1226, i64 %1362
  store i32 %1371, ptr %1372, align 4, !tbaa !1
  %1373 = add i64 %1362, 1
  br label %1361

1374:                                             ; preds = %1361
  br label %1375

1375:                                             ; preds = %1378, %1374
  %1376 = phi i64 [ %1385, %1378 ], [ 0, %1374 ]
  %1377 = icmp slt i64 %1376, 2
  br i1 %1377, label %1378, label %1386

1378:                                             ; preds = %1375
  %1379 = getelementptr inbounds i32, ptr %1109, i64 %1376
  %1380 = load i32, ptr %1379, align 4, !tbaa !1
  %1381 = getelementptr inbounds i32, ptr %1226, i64 %1376
  %1382 = load i32, ptr %1381, align 4, !tbaa !1
  %1383 = or i32 %1380, %1382
  %1384 = getelementptr inbounds i32, ptr %1226, i64 %1376
  store i32 %1383, ptr %1384, align 4, !tbaa !1
  %1385 = add i64 %1376, 1
  br label %1375

1386:                                             ; preds = %1375
  br label %1387

1387:                                             ; preds = %1390, %1386
  %1388 = phi i64 [ %1397, %1390 ], [ 0, %1386 ]
  %1389 = icmp slt i64 %1388, 2
  br i1 %1389, label %1390, label %1398

1390:                                             ; preds = %1387
  %1391 = getelementptr inbounds i32, ptr %1103, i64 %1388
  %1392 = load i32, ptr %1391, align 4, !tbaa !1
  %1393 = getelementptr inbounds i32, ptr %1226, i64 %1388
  %1394 = load i32, ptr %1393, align 4, !tbaa !1
  %1395 = xor i32 %1392, %1394
  %1396 = getelementptr inbounds i32, ptr %1121, i64 %1388
  store i32 %1395, ptr %1396, align 4, !tbaa !1
  %1397 = add i64 %1388, 1
  br label %1387

1398:                                             ; preds = %1387
  %1399 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1214, 1
  %1400 = getelementptr inbounds i32, ptr %1399, i32 2
  %1401 = load i32, ptr %1400, align 4, !tbaa !1
  br label %1402

1402:                                             ; preds = %1405, %1398
  %1403 = phi i64 [ %1412, %1405 ], [ 0, %1398 ]
  %1404 = icmp slt i64 %1403, 2
  br i1 %1404, label %1405, label %1413

1405:                                             ; preds = %1402
  %1406 = getelementptr inbounds i32, ptr %1103, i64 %1403
  %1407 = load i32, ptr %1406, align 4, !tbaa !1
  %1408 = getelementptr inbounds i32, ptr %1121, i64 %1403
  %1409 = load i32, ptr %1408, align 4, !tbaa !1
  %1410 = add i32 %1407, %1409
  %1411 = getelementptr inbounds i32, ptr %1127, i64 %1403
  store i32 %1410, ptr %1411, align 4, !tbaa !1
  %1412 = add i64 %1403, 1
  br label %1402

1413:                                             ; preds = %1402
  br label %1414

1414:                                             ; preds = %1417, %1413
  %1415 = phi i64 [ %1421, %1417 ], [ 0, %1413 ]
  %1416 = icmp slt i64 %1415, 2
  br i1 %1416, label %1417, label %1422

1417:                                             ; preds = %1414
  %1418 = getelementptr inbounds i32, ptr %1399, i32 2
  %1419 = load i32, ptr %1418, align 4, !tbaa !1
  %1420 = getelementptr inbounds i32, ptr %1226, i64 %1415
  store i32 %1419, ptr %1420, align 4, !tbaa !1
  %1421 = add i64 %1415, 1
  br label %1414

1422:                                             ; preds = %1414
  br label %1423

1423:                                             ; preds = %1426, %1422
  %1424 = phi i64 [ %1435, %1426 ], [ 0, %1422 ]
  %1425 = icmp slt i64 %1424, 2
  br i1 %1425, label %1426, label %1436

1426:                                             ; preds = %1423
  %1427 = getelementptr inbounds i32, ptr %1121, i64 %1424
  %1428 = load i32, ptr %1427, align 4, !tbaa !1
  %1429 = getelementptr inbounds i32, ptr %1226, i64 %1424
  %1430 = load i32, ptr %1429, align 4, !tbaa !1
  %1431 = shl i32 %1428, %1430
  %1432 = icmp ult i32 %1430, 32
  %1433 = select i1 %1432, i32 %1431, i32 0
  %1434 = getelementptr inbounds i32, ptr %1133, i64 %1424
  store i32 %1433, ptr %1434, align 4, !tbaa !1
  %1435 = add i64 %1424, 1
  br label %1423

1436:                                             ; preds = %1423
  %1437 = sub i32 32, %1401
  store i32 %1437, ptr %1139, align 4, !tbaa !1
  br label %1438

1438:                                             ; preds = %1441, %1436
  %1439 = phi i64 [ %1444, %1441 ], [ 0, %1436 ]
  %1440 = icmp slt i64 %1439, 2
  br i1 %1440, label %1441, label %1445

1441:                                             ; preds = %1438
  %1442 = load i32, ptr %1139, align 4, !tbaa !1
  %1443 = getelementptr inbounds i32, ptr %1226, i64 %1439
  store i32 %1442, ptr %1443, align 4, !tbaa !1
  %1444 = add i64 %1439, 1
  br label %1438

1445:                                             ; preds = %1438
  br label %1446

1446:                                             ; preds = %1449, %1445
  %1447 = phi i64 [ %1458, %1449 ], [ 0, %1445 ]
  %1448 = icmp slt i64 %1447, 2
  br i1 %1448, label %1449, label %1459

1449:                                             ; preds = %1446
  %1450 = getelementptr inbounds i32, ptr %1121, i64 %1447
  %1451 = load i32, ptr %1450, align 4, !tbaa !1
  %1452 = getelementptr inbounds i32, ptr %1226, i64 %1447
  %1453 = load i32, ptr %1452, align 4, !tbaa !1
  %1454 = lshr i32 %1451, %1453
  %1455 = icmp ult i32 %1453, 32
  %1456 = select i1 %1455, i32 %1454, i32 0
  %1457 = getelementptr inbounds i32, ptr %1226, i64 %1447
  store i32 %1456, ptr %1457, align 4, !tbaa !1
  %1458 = add i64 %1447, 1
  br label %1446

1459:                                             ; preds = %1446
  br label %1460

1460:                                             ; preds = %1463, %1459
  %1461 = phi i64 [ %1470, %1463 ], [ 0, %1459 ]
  %1462 = icmp slt i64 %1461, 2
  br i1 %1462, label %1463, label %1471

1463:                                             ; preds = %1460
  %1464 = getelementptr inbounds i32, ptr %1133, i64 %1461
  %1465 = load i32, ptr %1464, align 4, !tbaa !1
  %1466 = getelementptr inbounds i32, ptr %1226, i64 %1461
  %1467 = load i32, ptr %1466, align 4, !tbaa !1
  %1468 = or i32 %1465, %1467
  %1469 = getelementptr inbounds i32, ptr %1226, i64 %1461
  store i32 %1468, ptr %1469, align 4, !tbaa !1
  %1470 = add i64 %1461, 1
  br label %1460

1471:                                             ; preds = %1460
  br label %1472

1472:                                             ; preds = %1475, %1471
  %1473 = phi i64 [ %1482, %1475 ], [ 0, %1471 ]
  %1474 = icmp slt i64 %1473, 2
  br i1 %1474, label %1475, label %1483

1475:                                             ; preds = %1472
  %1476 = getelementptr inbounds i32, ptr %1127, i64 %1473
  %1477 = load i32, ptr %1476, align 4, !tbaa !1
  %1478 = getelementptr inbounds i32, ptr %1226, i64 %1473
  %1479 = load i32, ptr %1478, align 4, !tbaa !1
  %1480 = xor i32 %1477, %1479
  %1481 = getelementptr inbounds i32, ptr %1145, i64 %1473
  store i32 %1480, ptr %1481, align 4, !tbaa !1
  %1482 = add i64 %1473, 1
  br label %1472

1483:                                             ; preds = %1472
  %1484 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1214, 1
  %1485 = getelementptr inbounds i32, ptr %1484, i32 3
  %1486 = load i32, ptr %1485, align 4, !tbaa !1
  br label %1487

1487:                                             ; preds = %1490, %1483
  %1488 = phi i64 [ %1497, %1490 ], [ 0, %1483 ]
  %1489 = icmp slt i64 %1488, 2
  br i1 %1489, label %1490, label %1498

1490:                                             ; preds = %1487
  %1491 = getelementptr inbounds i32, ptr %1127, i64 %1488
  %1492 = load i32, ptr %1491, align 4, !tbaa !1
  %1493 = getelementptr inbounds i32, ptr %1145, i64 %1488
  %1494 = load i32, ptr %1493, align 4, !tbaa !1
  %1495 = add i32 %1492, %1494
  %1496 = getelementptr inbounds i32, ptr %1151, i64 %1488
  store i32 %1495, ptr %1496, align 4, !tbaa !1
  %1497 = add i64 %1488, 1
  br label %1487

1498:                                             ; preds = %1487
  br label %1499

1499:                                             ; preds = %1502, %1498
  %1500 = phi i64 [ %1506, %1502 ], [ 0, %1498 ]
  %1501 = icmp slt i64 %1500, 2
  br i1 %1501, label %1502, label %1507

1502:                                             ; preds = %1499
  %1503 = getelementptr inbounds i32, ptr %1484, i32 3
  %1504 = load i32, ptr %1503, align 4, !tbaa !1
  %1505 = getelementptr inbounds i32, ptr %1226, i64 %1500
  store i32 %1504, ptr %1505, align 4, !tbaa !1
  %1506 = add i64 %1500, 1
  br label %1499

1507:                                             ; preds = %1499
  br label %1508

1508:                                             ; preds = %1511, %1507
  %1509 = phi i64 [ %1520, %1511 ], [ 0, %1507 ]
  %1510 = icmp slt i64 %1509, 2
  br i1 %1510, label %1511, label %1521

1511:                                             ; preds = %1508
  %1512 = getelementptr inbounds i32, ptr %1145, i64 %1509
  %1513 = load i32, ptr %1512, align 4, !tbaa !1
  %1514 = getelementptr inbounds i32, ptr %1226, i64 %1509
  %1515 = load i32, ptr %1514, align 4, !tbaa !1
  %1516 = shl i32 %1513, %1515
  %1517 = icmp ult i32 %1515, 32
  %1518 = select i1 %1517, i32 %1516, i32 0
  %1519 = getelementptr inbounds i32, ptr %1157, i64 %1509
  store i32 %1518, ptr %1519, align 4, !tbaa !1
  %1520 = add i64 %1509, 1
  br label %1508

1521:                                             ; preds = %1508
  %1522 = sub i32 32, %1486
  store i32 %1522, ptr %1163, align 4, !tbaa !1
  br label %1523

1523:                                             ; preds = %1526, %1521
  %1524 = phi i64 [ %1529, %1526 ], [ 0, %1521 ]
  %1525 = icmp slt i64 %1524, 2
  br i1 %1525, label %1526, label %1530

1526:                                             ; preds = %1523
  %1527 = load i32, ptr %1163, align 4, !tbaa !1
  %1528 = getelementptr inbounds i32, ptr %1226, i64 %1524
  store i32 %1527, ptr %1528, align 4, !tbaa !1
  %1529 = add i64 %1524, 1
  br label %1523

1530:                                             ; preds = %1523
  br label %1531

1531:                                             ; preds = %1534, %1530
  %1532 = phi i64 [ %1543, %1534 ], [ 0, %1530 ]
  %1533 = icmp slt i64 %1532, 2
  br i1 %1533, label %1534, label %1544

1534:                                             ; preds = %1531
  %1535 = getelementptr inbounds i32, ptr %1145, i64 %1532
  %1536 = load i32, ptr %1535, align 4, !tbaa !1
  %1537 = getelementptr inbounds i32, ptr %1226, i64 %1532
  %1538 = load i32, ptr %1537, align 4, !tbaa !1
  %1539 = lshr i32 %1536, %1538
  %1540 = icmp ult i32 %1538, 32
  %1541 = select i1 %1540, i32 %1539, i32 0
  %1542 = getelementptr inbounds i32, ptr %1226, i64 %1532
  store i32 %1541, ptr %1542, align 4, !tbaa !1
  %1543 = add i64 %1532, 1
  br label %1531

1544:                                             ; preds = %1531
  br label %1545

1545:                                             ; preds = %1548, %1544
  %1546 = phi i64 [ %1555, %1548 ], [ 0, %1544 ]
  %1547 = icmp slt i64 %1546, 2
  br i1 %1547, label %1548, label %1556

1548:                                             ; preds = %1545
  %1549 = getelementptr inbounds i32, ptr %1157, i64 %1546
  %1550 = load i32, ptr %1549, align 4, !tbaa !1
  %1551 = getelementptr inbounds i32, ptr %1226, i64 %1546
  %1552 = load i32, ptr %1551, align 4, !tbaa !1
  %1553 = or i32 %1550, %1552
  %1554 = getelementptr inbounds i32, ptr %1226, i64 %1546
  store i32 %1553, ptr %1554, align 4, !tbaa !1
  %1555 = add i64 %1546, 1
  br label %1545

1556:                                             ; preds = %1545
  br label %1557

1557:                                             ; preds = %1560, %1556
  %1558 = phi i64 [ %1567, %1560 ], [ 0, %1556 ]
  %1559 = icmp slt i64 %1558, 2
  br i1 %1559, label %1560, label %1568

1560:                                             ; preds = %1557
  %1561 = getelementptr inbounds i32, ptr %1151, i64 %1558
  %1562 = load i32, ptr %1561, align 4, !tbaa !1
  %1563 = getelementptr inbounds i32, ptr %1226, i64 %1558
  %1564 = load i32, ptr %1563, align 4, !tbaa !1
  %1565 = xor i32 %1562, %1564
  %1566 = getelementptr inbounds i32, ptr %1169, i64 %1558
  store i32 %1565, ptr %1566, align 4, !tbaa !1
  %1567 = add i64 %1558, 1
  br label %1557

1568:                                             ; preds = %1557
  br label %1569

1569:                                             ; preds = %1572, %1568
  %1570 = phi i64 [ %1575, %1572 ], [ 0, %1568 ]
  %1571 = icmp slt i64 %1570, 2
  br i1 %1571, label %1572, label %1576

1572:                                             ; preds = %1569
  %1573 = load i32, ptr %1067, align 4, !tbaa !1
  %1574 = getelementptr inbounds i32, ptr %1226, i64 %1570
  store i32 %1573, ptr %1574, align 4, !tbaa !1
  %1575 = add i64 %1570, 1
  br label %1569

1576:                                             ; preds = %1569
  %1577 = call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %1578 = ptrtoint ptr %1577 to i64
  %1579 = add i64 %1578, 63
  %1580 = urem i64 %1579, 64
  %1581 = sub i64 %1579, %1580
  %1582 = inttoptr i64 %1581 to ptr
  br label %1583

1583:                                             ; preds = %1586, %1576
  %1584 = phi i64 [ %1593, %1586 ], [ 0, %1576 ]
  %1585 = icmp slt i64 %1584, 2
  br i1 %1585, label %1586, label %1594

1586:                                             ; preds = %1583
  %1587 = getelementptr inbounds i32, ptr %1151, i64 %1584
  %1588 = load i32, ptr %1587, align 4, !tbaa !1
  %1589 = getelementptr inbounds i32, ptr %1226, i64 %1584
  %1590 = load i32, ptr %1589, align 4, !tbaa !1
  %1591 = add i32 %1588, %1590
  %1592 = getelementptr inbounds i32, ptr %1582, i64 %1584
  store i32 %1591, ptr %1592, align 4, !tbaa !1
  %1593 = add i64 %1584, 1
  br label %1583

1594:                                             ; preds = %1583
  br label %1595

1595:                                             ; preds = %1598, %1594
  %1596 = phi i64 [ %1601, %1598 ], [ 0, %1594 ]
  %1597 = icmp slt i64 %1596, 2
  br i1 %1597, label %1598, label %1602

1598:                                             ; preds = %1595
  %1599 = load i32, ptr %1073, align 4, !tbaa !1
  %1600 = getelementptr inbounds i32, ptr %1226, i64 %1596
  store i32 %1599, ptr %1600, align 4, !tbaa !1
  %1601 = add i64 %1596, 1
  br label %1595

1602:                                             ; preds = %1595
  br label %1603

1603:                                             ; preds = %1606, %1602
  %1604 = phi i64 [ %1613, %1606 ], [ 0, %1602 ]
  %1605 = icmp slt i64 %1604, 2
  br i1 %1605, label %1606, label %1614

1606:                                             ; preds = %1603
  %1607 = getelementptr inbounds i32, ptr %1169, i64 %1604
  %1608 = load i32, ptr %1607, align 4, !tbaa !1
  %1609 = getelementptr inbounds i32, ptr %1226, i64 %1604
  %1610 = load i32, ptr %1609, align 4, !tbaa !1
  %1611 = add i32 %1608, %1610
  %1612 = getelementptr inbounds i32, ptr %1175, i64 %1604
  store i32 %1611, ptr %1612, align 4, !tbaa !1
  %1613 = add i64 %1604, 1
  br label %1603

1614:                                             ; preds = %1603
  %1615 = trunc i64 %1218 to i32
  store i32 %1615, ptr %1181, align 4, !tbaa !1
  br label %1616

1616:                                             ; preds = %1619, %1614
  %1617 = phi i64 [ %1622, %1619 ], [ 0, %1614 ]
  %1618 = icmp slt i64 %1617, 2
  br i1 %1618, label %1619, label %1623

1619:                                             ; preds = %1616
  %1620 = load i32, ptr %1181, align 4, !tbaa !1
  %1621 = getelementptr inbounds i32, ptr %1226, i64 %1617
  store i32 %1620, ptr %1621, align 4, !tbaa !1
  %1622 = add i64 %1617, 1
  br label %1616

1623:                                             ; preds = %1616
  br label %1624

1624:                                             ; preds = %1627, %1623
  %1625 = phi i64 [ %1634, %1627 ], [ 0, %1623 ]
  %1626 = icmp slt i64 %1625, 2
  br i1 %1626, label %1627, label %1635

1627:                                             ; preds = %1624
  %1628 = getelementptr inbounds i32, ptr %1175, i64 %1625
  %1629 = load i32, ptr %1628, align 4, !tbaa !1
  %1630 = getelementptr inbounds i32, ptr %1226, i64 %1625
  %1631 = load i32, ptr %1630, align 4, !tbaa !1
  %1632 = add i32 %1629, %1631
  %1633 = getelementptr inbounds i32, ptr %1226, i64 %1625
  store i32 %1632, ptr %1633, align 4, !tbaa !1
  %1634 = add i64 %1625, 1
  br label %1624

1635:                                             ; preds = %1624
  %1636 = call ptr @_mlir_memref_to_llvm_alloc(i64 80)
  %1637 = ptrtoint ptr %1636 to i64
  %1638 = add i64 %1637, 63
  %1639 = urem i64 %1638, 64
  %1640 = sub i64 %1638, %1639
  %1641 = inttoptr i64 %1640 to ptr
  %1642 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1215, 3, 0
  %1643 = mul i64 %1642, 1
  %1644 = mul i64 %1643, 4
  %1645 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1215, 1
  %1646 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1215, 2
  %1647 = getelementptr inbounds i32, ptr %1645, i64 %1646
  call void @llvm.memcpy.p0.p0.i64(ptr %1641, ptr %1647, i64 %1644, i1 false)
  %1648 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1215, 0
  call void @_mlir_memref_to_llvm_free(ptr %1648)
  %1649 = call ptr @_mlir_memref_to_llvm_alloc(i64 80)
  %1650 = ptrtoint ptr %1649 to i64
  %1651 = add i64 %1650, 63
  %1652 = urem i64 %1651, 64
  %1653 = sub i64 %1651, %1652
  %1654 = inttoptr i64 %1653 to ptr
  %1655 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1214, 3, 0
  %1656 = mul i64 %1655, 1
  %1657 = mul i64 %1656, 4
  %1658 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1214, 1
  %1659 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1214, 2
  %1660 = getelementptr inbounds i32, ptr %1658, i64 %1659
  call void @llvm.memcpy.p0.p0.i64(ptr %1654, ptr %1660, i64 %1657, i1 false)
  %1661 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1214, 0
  call void @_mlir_memref_to_llvm_free(ptr %1661)
  %1662 = call ptr @_mlir_memref_to_llvm_alloc(i64 8)
  %1663 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %1662, 0
  %1664 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1663, ptr %1662, 1
  %1665 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1664, i64 0, 2
  %1666 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1665, i64 2, 3, 0
  %1667 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1666, i64 1, 4, 0
  call void @llvm.memcpy.p0.p0.i64(ptr %1662, ptr %1226, i64 8, i1 false)
  call void @_mlir_memref_to_llvm_free(ptr %1221)
  %1668 = call ptr @_mlir_memref_to_llvm_alloc(i64 8)
  %1669 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %1668, 0
  %1670 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1669, ptr %1668, 1
  %1671 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1670, i64 0, 2
  %1672 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1671, i64 2, 3, 0
  %1673 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1672, i64 1, 4, 0
  call void @llvm.memcpy.p0.p0.i64(ptr %1668, ptr %1582, i64 8, i1 false)
  call void @_mlir_memref_to_llvm_free(ptr %1577)
  %1674 = call ptr @_mlir_memref_to_llvm_alloc(i64 16)
  %1675 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %1674, 0
  %1676 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1675, ptr %1674, 1
  %1677 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1676, i64 0, 2
  %1678 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1677, i64 4, 3, 0
  %1679 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1678, i64 1, 4, 0
  call void @llvm.memcpy.p0.p0.i64(ptr %1674, ptr %1641, i64 16, i1 false)
  call void @_mlir_memref_to_llvm_free(ptr %1636)
  %1680 = call ptr @_mlir_memref_to_llvm_alloc(i64 16)
  %1681 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %1680, 0
  %1682 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1681, ptr %1680, 1
  %1683 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1682, i64 0, 2
  %1684 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1683, i64 4, 3, 0
  %1685 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1684, i64 1, 4, 0
  call void @llvm.memcpy.p0.p0.i64(ptr %1680, ptr %1654, i64 16, i1 false)
  call void @_mlir_memref_to_llvm_free(ptr %1649)
  %1686 = add i64 %1207, 1
  br label %1206

1687:                                             ; preds = %1206
  %1688 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1215, 0
  call void @_mlir_memref_to_llvm_free(ptr %1688)
  %1689 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1214, 0
  call void @_mlir_memref_to_llvm_free(ptr %1689)
  call void @_mlir_memref_to_llvm_free(ptr %1176)
  call void @_mlir_memref_to_llvm_free(ptr %1170)
  call void @_mlir_memref_to_llvm_free(ptr %1164)
  call void @_mlir_memref_to_llvm_free(ptr %1158)
  call void @_mlir_memref_to_llvm_free(ptr %1152)
  call void @_mlir_memref_to_llvm_free(ptr %1146)
  call void @_mlir_memref_to_llvm_free(ptr %1140)
  call void @_mlir_memref_to_llvm_free(ptr %1134)
  call void @_mlir_memref_to_llvm_free(ptr %1128)
  call void @_mlir_memref_to_llvm_free(ptr %1122)
  call void @_mlir_memref_to_llvm_free(ptr %1116)
  call void @_mlir_memref_to_llvm_free(ptr %1110)
  call void @_mlir_memref_to_llvm_free(ptr %1104)
  call void @_mlir_memref_to_llvm_free(ptr %1098)
  call void @_mlir_memref_to_llvm_free(ptr %1092)
  call void @_mlir_memref_to_llvm_free(ptr %1086)
  call void @_mlir_memref_to_llvm_free(ptr %1080)
  call void @_mlir_memref_to_llvm_free(ptr %1074)
  call void @_mlir_memref_to_llvm_free(ptr %1068)
  call void @_mlir_memref_to_llvm_free(ptr %1062)
  %1690 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1209, 1
  %1691 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1210, 1
  %1692 = call ptr @_mlir_memref_to_llvm_alloc(i64 80)
  %1693 = ptrtoint ptr %1692 to i64
  %1694 = add i64 %1693, 63
  %1695 = urem i64 %1694, 64
  %1696 = sub i64 %1694, %1695
  %1697 = inttoptr i64 %1696 to ptr
  br label %1698

1698:                                             ; preds = %1723, %1687
  %1699 = phi i64 [ %1724, %1723 ], [ 0, %1687 ]
  %1700 = icmp slt i64 %1699, 2
  br i1 %1700, label %1701, label %1725

1701:                                             ; preds = %1698
  br label %1702

1702:                                             ; preds = %1718, %1701
  %1703 = phi i64 [ %1722, %1718 ], [ 0, %1701 ]
  %1704 = icmp slt i64 %1703, 2
  br i1 %1704, label %1705, label %1723

1705:                                             ; preds = %1702
  %1706 = icmp ult i64 %1703, 1
  br i1 %1706, label %1707, label %1711

1707:                                             ; preds = %1705
  %1708 = add i64 %1699, %1703
  %1709 = getelementptr inbounds i32, ptr %1690, i64 %1708
  %1710 = load i32, ptr %1709, align 4, !tbaa !1
  br label %1716

1711:                                             ; preds = %1705
  %1712 = sub i64 %1703, 1
  %1713 = add i64 %1699, %1712
  %1714 = getelementptr inbounds i32, ptr %1691, i64 %1713
  %1715 = load i32, ptr %1714, align 4, !tbaa !1
  br label %1716

1716:                                             ; preds = %1707, %1711
  %1717 = phi i32 [ %1715, %1711 ], [ %1710, %1707 ]
  br label %1718

1718:                                             ; preds = %1716
  %1719 = mul i64 %1699, 2
  %1720 = add i64 %1719, %1703
  %1721 = getelementptr inbounds i32, ptr %1697, i64 %1720
  store i32 %1717, ptr %1721, align 4, !tbaa !1
  %1722 = add i64 %1703, 1
  br label %1702

1723:                                             ; preds = %1702
  %1724 = add i64 %1699, 1
  br label %1698

1725:                                             ; preds = %1698
  %1726 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1209, 0
  call void @_mlir_memref_to_llvm_free(ptr %1726)
  %1727 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1210, 0
  call void @_mlir_memref_to_llvm_free(ptr %1727)
  %1728 = load i32, ptr %1697, align 4, !tbaa !1
  %1729 = getelementptr inbounds i32, ptr %1697, i32 1
  %1730 = load i32, ptr %1729, align 4, !tbaa !1
  %1731 = xor i32 %1728, %1730
  %1732 = xor i32 %1731, 466688986
  %1733 = load i32, ptr %1697, align 4, !tbaa !1
  %1734 = getelementptr inbounds i32, ptr %1697, i32 1
  %1735 = load i32, ptr %1734, align 4, !tbaa !1
  %1736 = getelementptr inbounds i32, ptr %1697, i32 1
  %1737 = load i32, ptr %1736, align 4, !tbaa !1
  %1738 = load i32, ptr %1697, align 4, !tbaa !1
  %1739 = call ptr @_mlir_memref_to_llvm_alloc(i64 80)
  %1740 = ptrtoint ptr %1739 to i64
  %1741 = add i64 %1740, 63
  %1742 = urem i64 %1741, 64
  %1743 = sub i64 %1741, %1742
  %1744 = inttoptr i64 %1743 to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr %1744, ptr @__constant_4xi32_3, i64 16, i1 false)
  %1745 = call ptr @_mlir_memref_to_llvm_alloc(i64 80)
  %1746 = ptrtoint ptr %1745 to i64
  %1747 = add i64 %1746, 63
  %1748 = urem i64 %1747, 64
  %1749 = sub i64 %1747, %1748
  %1750 = inttoptr i64 %1749 to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr %1750, ptr @__constant_4xi32, i64 16, i1 false)
  %1751 = call ptr @_mlir_memref_to_llvm_alloc(i64 16)
  %1752 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %1751, 0
  %1753 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1752, ptr %1751, 1
  %1754 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1753, i64 0, 2
  %1755 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1754, i64 4, 3, 0
  %1756 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1755, i64 1, 4, 0
  call void @llvm.memcpy.p0.p0.i64(ptr %1751, ptr %1744, i64 16, i1 false)
  call void @_mlir_memref_to_llvm_free(ptr %1739)
  %1757 = call ptr @_mlir_memref_to_llvm_alloc(i64 16)
  %1758 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %1757, 0
  %1759 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1758, ptr %1757, 1
  %1760 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1759, i64 0, 2
  %1761 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1760, i64 4, 3, 0
  %1762 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1761, i64 1, 4, 0
  call void @llvm.memcpy.p0.p0.i64(ptr %1757, ptr %1750, i64 16, i1 false)
  call void @_mlir_memref_to_llvm_free(ptr %1745)
  br label %1763

1763:                                             ; preds = %1774, %1725
  %1764 = phi i64 [ %1869, %1774 ], [ 0, %1725 ]
  %1765 = phi i64 [ %1775, %1774 ], [ 0, %1725 ]
  %1766 = phi i32 [ %1827, %1774 ], [ %1733, %1725 ]
  %1767 = phi i32 [ %1830, %1774 ], [ %1735, %1725 ]
  %1768 = phi i32 [ %1769, %1774 ], [ %1737, %1725 ]
  %1769 = phi i32 [ %1770, %1774 ], [ %1732, %1725 ]
  %1770 = phi i32 [ %1768, %1774 ], [ %1738, %1725 ]
  %1771 = phi { ptr, ptr, i64, [1 x i64], [1 x i64] } [ %1862, %1774 ], [ %1756, %1725 ]
  %1772 = phi { ptr, ptr, i64, [1 x i64], [1 x i64] } [ %1868, %1774 ], [ %1762, %1725 ]
  %1773 = icmp slt i64 %1764, 5
  br i1 %1773, label %1774, label %1870

1774:                                             ; preds = %1763
  %1775 = add i64 %1765, 1
  %1776 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1771, 1
  %1777 = load i32, ptr %1776, align 4, !tbaa !1
  %1778 = add i32 %1766, %1767
  %1779 = shl i32 %1767, %1777
  %1780 = icmp ult i32 %1777, 32
  %1781 = select i1 %1780, i32 %1779, i32 0
  %1782 = sub i32 32, %1777
  %1783 = lshr i32 %1767, %1782
  %1784 = icmp ult i32 %1782, 32
  %1785 = select i1 %1784, i32 %1783, i32 0
  %1786 = or i32 %1781, %1785
  %1787 = xor i32 %1778, %1786
  %1788 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1771, 1
  %1789 = getelementptr inbounds i32, ptr %1788, i32 1
  %1790 = load i32, ptr %1789, align 4, !tbaa !1
  %1791 = add i32 %1778, %1787
  %1792 = shl i32 %1787, %1790
  %1793 = icmp ult i32 %1790, 32
  %1794 = select i1 %1793, i32 %1792, i32 0
  %1795 = sub i32 32, %1790
  %1796 = lshr i32 %1787, %1795
  %1797 = icmp ult i32 %1795, 32
  %1798 = select i1 %1797, i32 %1796, i32 0
  %1799 = or i32 %1794, %1798
  %1800 = xor i32 %1791, %1799
  %1801 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1771, 1
  %1802 = getelementptr inbounds i32, ptr %1801, i32 2
  %1803 = load i32, ptr %1802, align 4, !tbaa !1
  %1804 = add i32 %1791, %1800
  %1805 = shl i32 %1800, %1803
  %1806 = icmp ult i32 %1803, 32
  %1807 = select i1 %1806, i32 %1805, i32 0
  %1808 = sub i32 32, %1803
  %1809 = lshr i32 %1800, %1808
  %1810 = icmp ult i32 %1808, 32
  %1811 = select i1 %1810, i32 %1809, i32 0
  %1812 = or i32 %1807, %1811
  %1813 = xor i32 %1804, %1812
  %1814 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1771, 1
  %1815 = getelementptr inbounds i32, ptr %1814, i32 3
  %1816 = load i32, ptr %1815, align 4, !tbaa !1
  %1817 = add i32 %1804, %1813
  %1818 = shl i32 %1813, %1816
  %1819 = icmp ult i32 %1816, 32
  %1820 = select i1 %1819, i32 %1818, i32 0
  %1821 = sub i32 32, %1816
  %1822 = lshr i32 %1813, %1821
  %1823 = icmp ult i32 %1821, 32
  %1824 = select i1 %1823, i32 %1822, i32 0
  %1825 = or i32 %1820, %1824
  %1826 = xor i32 %1817, %1825
  %1827 = add i32 %1817, %1768
  %1828 = add i32 %1826, %1769
  %1829 = trunc i64 %1775 to i32
  %1830 = add i32 %1828, %1829
  %1831 = call ptr @_mlir_memref_to_llvm_alloc(i64 80)
  %1832 = ptrtoint ptr %1831 to i64
  %1833 = add i64 %1832, 63
  %1834 = urem i64 %1833, 64
  %1835 = sub i64 %1833, %1834
  %1836 = inttoptr i64 %1835 to ptr
  %1837 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1772, 3, 0
  %1838 = mul i64 %1837, 1
  %1839 = mul i64 %1838, 4
  %1840 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1772, 1
  %1841 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1772, 2
  %1842 = getelementptr inbounds i32, ptr %1840, i64 %1841
  call void @llvm.memcpy.p0.p0.i64(ptr %1836, ptr %1842, i64 %1839, i1 false)
  %1843 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1772, 0
  call void @_mlir_memref_to_llvm_free(ptr %1843)
  %1844 = call ptr @_mlir_memref_to_llvm_alloc(i64 80)
  %1845 = ptrtoint ptr %1844 to i64
  %1846 = add i64 %1845, 63
  %1847 = urem i64 %1846, 64
  %1848 = sub i64 %1846, %1847
  %1849 = inttoptr i64 %1848 to ptr
  %1850 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1771, 3, 0
  %1851 = mul i64 %1850, 1
  %1852 = mul i64 %1851, 4
  %1853 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1771, 1
  %1854 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1771, 2
  %1855 = getelementptr inbounds i32, ptr %1853, i64 %1854
  call void @llvm.memcpy.p0.p0.i64(ptr %1849, ptr %1855, i64 %1852, i1 false)
  %1856 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1771, 0
  call void @_mlir_memref_to_llvm_free(ptr %1856)
  %1857 = call ptr @_mlir_memref_to_llvm_alloc(i64 16)
  %1858 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %1857, 0
  %1859 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1858, ptr %1857, 1
  %1860 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1859, i64 0, 2
  %1861 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1860, i64 4, 3, 0
  %1862 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1861, i64 1, 4, 0
  call void @llvm.memcpy.p0.p0.i64(ptr %1857, ptr %1836, i64 16, i1 false)
  call void @_mlir_memref_to_llvm_free(ptr %1831)
  %1863 = call ptr @_mlir_memref_to_llvm_alloc(i64 16)
  %1864 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %1863, 0
  %1865 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1864, ptr %1863, 1
  %1866 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1865, i64 0, 2
  %1867 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1866, i64 4, 3, 0
  %1868 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1867, i64 1, 4, 0
  call void @llvm.memcpy.p0.p0.i64(ptr %1863, ptr %1849, i64 16, i1 false)
  call void @_mlir_memref_to_llvm_free(ptr %1844)
  %1869 = add i64 %1764, 1
  br label %1763

1870:                                             ; preds = %1763
  %1871 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1772, 0
  call void @_mlir_memref_to_llvm_free(ptr %1871)
  %1872 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1771, 0
  call void @_mlir_memref_to_llvm_free(ptr %1872)
  %1873 = zext i32 %1766 to i64
  %1874 = zext i32 %1767 to i64
  %1875 = shl i64 %1873, 32
  %1876 = or i64 %1875, %1874
  %1877 = getelementptr inbounds i32, ptr %1697, i32 2
  %1878 = load i32, ptr %1877, align 4, !tbaa !1
  %1879 = getelementptr inbounds i32, ptr %1697, i32 3
  %1880 = load i32, ptr %1879, align 4, !tbaa !1
  %1881 = xor i32 %1878, %1880
  %1882 = xor i32 %1881, 466688986
  %1883 = getelementptr inbounds i32, ptr %1697, i32 2
  %1884 = load i32, ptr %1883, align 4, !tbaa !1
  %1885 = getelementptr inbounds i32, ptr %1697, i32 3
  %1886 = load i32, ptr %1885, align 4, !tbaa !1
  %1887 = getelementptr inbounds i32, ptr %1697, i32 3
  %1888 = load i32, ptr %1887, align 4, !tbaa !1
  %1889 = getelementptr inbounds i32, ptr %1697, i32 2
  %1890 = load i32, ptr %1889, align 4, !tbaa !1
  call void @_mlir_memref_to_llvm_free(ptr %1692)
  %1891 = call ptr @_mlir_memref_to_llvm_alloc(i64 80)
  %1892 = ptrtoint ptr %1891 to i64
  %1893 = add i64 %1892, 63
  %1894 = urem i64 %1893, 64
  %1895 = sub i64 %1893, %1894
  %1896 = inttoptr i64 %1895 to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr %1896, ptr @__constant_4xi32_3, i64 16, i1 false)
  %1897 = call ptr @_mlir_memref_to_llvm_alloc(i64 80)
  %1898 = ptrtoint ptr %1897 to i64
  %1899 = add i64 %1898, 63
  %1900 = urem i64 %1899, 64
  %1901 = sub i64 %1899, %1900
  %1902 = inttoptr i64 %1901 to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr %1902, ptr @__constant_4xi32, i64 16, i1 false)
  %1903 = call ptr @_mlir_memref_to_llvm_alloc(i64 16)
  %1904 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %1903, 0
  %1905 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1904, ptr %1903, 1
  %1906 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1905, i64 0, 2
  %1907 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1906, i64 4, 3, 0
  %1908 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1907, i64 1, 4, 0
  call void @llvm.memcpy.p0.p0.i64(ptr %1903, ptr %1896, i64 16, i1 false)
  call void @_mlir_memref_to_llvm_free(ptr %1891)
  %1909 = call ptr @_mlir_memref_to_llvm_alloc(i64 16)
  %1910 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %1909, 0
  %1911 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1910, ptr %1909, 1
  %1912 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1911, i64 0, 2
  %1913 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1912, i64 4, 3, 0
  %1914 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1913, i64 1, 4, 0
  call void @llvm.memcpy.p0.p0.i64(ptr %1909, ptr %1902, i64 16, i1 false)
  call void @_mlir_memref_to_llvm_free(ptr %1897)
  br label %1915

1915:                                             ; preds = %1926, %1870
  %1916 = phi i64 [ %2021, %1926 ], [ 0, %1870 ]
  %1917 = phi i64 [ %1927, %1926 ], [ 0, %1870 ]
  %1918 = phi i32 [ %1979, %1926 ], [ %1884, %1870 ]
  %1919 = phi i32 [ %1982, %1926 ], [ %1886, %1870 ]
  %1920 = phi i32 [ %1921, %1926 ], [ %1888, %1870 ]
  %1921 = phi i32 [ %1922, %1926 ], [ %1882, %1870 ]
  %1922 = phi i32 [ %1920, %1926 ], [ %1890, %1870 ]
  %1923 = phi { ptr, ptr, i64, [1 x i64], [1 x i64] } [ %2014, %1926 ], [ %1908, %1870 ]
  %1924 = phi { ptr, ptr, i64, [1 x i64], [1 x i64] } [ %2020, %1926 ], [ %1914, %1870 ]
  %1925 = icmp slt i64 %1916, 5
  br i1 %1925, label %1926, label %2022

1926:                                             ; preds = %1915
  %1927 = add i64 %1917, 1
  %1928 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1923, 1
  %1929 = load i32, ptr %1928, align 4, !tbaa !1
  %1930 = add i32 %1918, %1919
  %1931 = shl i32 %1919, %1929
  %1932 = icmp ult i32 %1929, 32
  %1933 = select i1 %1932, i32 %1931, i32 0
  %1934 = sub i32 32, %1929
  %1935 = lshr i32 %1919, %1934
  %1936 = icmp ult i32 %1934, 32
  %1937 = select i1 %1936, i32 %1935, i32 0
  %1938 = or i32 %1933, %1937
  %1939 = xor i32 %1930, %1938
  %1940 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1923, 1
  %1941 = getelementptr inbounds i32, ptr %1940, i32 1
  %1942 = load i32, ptr %1941, align 4, !tbaa !1
  %1943 = add i32 %1930, %1939
  %1944 = shl i32 %1939, %1942
  %1945 = icmp ult i32 %1942, 32
  %1946 = select i1 %1945, i32 %1944, i32 0
  %1947 = sub i32 32, %1942
  %1948 = lshr i32 %1939, %1947
  %1949 = icmp ult i32 %1947, 32
  %1950 = select i1 %1949, i32 %1948, i32 0
  %1951 = or i32 %1946, %1950
  %1952 = xor i32 %1943, %1951
  %1953 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1923, 1
  %1954 = getelementptr inbounds i32, ptr %1953, i32 2
  %1955 = load i32, ptr %1954, align 4, !tbaa !1
  %1956 = add i32 %1943, %1952
  %1957 = shl i32 %1952, %1955
  %1958 = icmp ult i32 %1955, 32
  %1959 = select i1 %1958, i32 %1957, i32 0
  %1960 = sub i32 32, %1955
  %1961 = lshr i32 %1952, %1960
  %1962 = icmp ult i32 %1960, 32
  %1963 = select i1 %1962, i32 %1961, i32 0
  %1964 = or i32 %1959, %1963
  %1965 = xor i32 %1956, %1964
  %1966 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1923, 1
  %1967 = getelementptr inbounds i32, ptr %1966, i32 3
  %1968 = load i32, ptr %1967, align 4, !tbaa !1
  %1969 = add i32 %1956, %1965
  %1970 = shl i32 %1965, %1968
  %1971 = icmp ult i32 %1968, 32
  %1972 = select i1 %1971, i32 %1970, i32 0
  %1973 = sub i32 32, %1968
  %1974 = lshr i32 %1965, %1973
  %1975 = icmp ult i32 %1973, 32
  %1976 = select i1 %1975, i32 %1974, i32 0
  %1977 = or i32 %1972, %1976
  %1978 = xor i32 %1969, %1977
  %1979 = add i32 %1969, %1920
  %1980 = add i32 %1978, %1921
  %1981 = trunc i64 %1927 to i32
  %1982 = add i32 %1980, %1981
  %1983 = call ptr @_mlir_memref_to_llvm_alloc(i64 80)
  %1984 = ptrtoint ptr %1983 to i64
  %1985 = add i64 %1984, 63
  %1986 = urem i64 %1985, 64
  %1987 = sub i64 %1985, %1986
  %1988 = inttoptr i64 %1987 to ptr
  %1989 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1924, 3, 0
  %1990 = mul i64 %1989, 1
  %1991 = mul i64 %1990, 4
  %1992 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1924, 1
  %1993 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1924, 2
  %1994 = getelementptr inbounds i32, ptr %1992, i64 %1993
  call void @llvm.memcpy.p0.p0.i64(ptr %1988, ptr %1994, i64 %1991, i1 false)
  %1995 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1924, 0
  call void @_mlir_memref_to_llvm_free(ptr %1995)
  %1996 = call ptr @_mlir_memref_to_llvm_alloc(i64 80)
  %1997 = ptrtoint ptr %1996 to i64
  %1998 = add i64 %1997, 63
  %1999 = urem i64 %1998, 64
  %2000 = sub i64 %1998, %1999
  %2001 = inttoptr i64 %2000 to ptr
  %2002 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1923, 3, 0
  %2003 = mul i64 %2002, 1
  %2004 = mul i64 %2003, 4
  %2005 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1923, 1
  %2006 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1923, 2
  %2007 = getelementptr inbounds i32, ptr %2005, i64 %2006
  call void @llvm.memcpy.p0.p0.i64(ptr %2001, ptr %2007, i64 %2004, i1 false)
  %2008 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1923, 0
  call void @_mlir_memref_to_llvm_free(ptr %2008)
  %2009 = call ptr @_mlir_memref_to_llvm_alloc(i64 16)
  %2010 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %2009, 0
  %2011 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %2010, ptr %2009, 1
  %2012 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %2011, i64 0, 2
  %2013 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %2012, i64 4, 3, 0
  %2014 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %2013, i64 1, 4, 0
  call void @llvm.memcpy.p0.p0.i64(ptr %2009, ptr %1988, i64 16, i1 false)
  call void @_mlir_memref_to_llvm_free(ptr %1983)
  %2015 = call ptr @_mlir_memref_to_llvm_alloc(i64 16)
  %2016 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %2015, 0
  %2017 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %2016, ptr %2015, 1
  %2018 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %2017, i64 0, 2
  %2019 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %2018, i64 4, 3, 0
  %2020 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %2019, i64 1, 4, 0
  call void @llvm.memcpy.p0.p0.i64(ptr %2015, ptr %2001, i64 16, i1 false)
  call void @_mlir_memref_to_llvm_free(ptr %1996)
  %2021 = add i64 %1916, 1
  br label %1915

2022:                                             ; preds = %1915
  %2023 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1924, 0
  call void @_mlir_memref_to_llvm_free(ptr %2023)
  %2024 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %1923, 0
  call void @_mlir_memref_to_llvm_free(ptr %2024)
  %2025 = zext i32 %1918 to i64
  %2026 = zext i32 %1919 to i64
  %2027 = shl i64 %2025, 32
  %2028 = or i64 %2027, %2026
  %2029 = urem i64 %1876, 96
  %2030 = mul i64 %2029, 64
  %2031 = urem i64 %2028, 96
  %2032 = add i64 %2030, %2031
  %2033 = urem i64 %2032, 96
  %2034 = load float, ptr %10, align 4, !tbaa !4
  %2035 = load float, ptr %13, align 4, !tbaa !4
  %2036 = call ptr @_mlir_memref_to_llvm_alloc(i64 68)
  %2037 = ptrtoint ptr %2036 to i64
  %2038 = add i64 %2037, 63
  %2039 = urem i64 %2038, 64
  %2040 = sub i64 %2038, %2039
  %2041 = inttoptr i64 %2040 to ptr
  %2042 = call ptr @_mlir_memref_to_llvm_alloc(i64 68)
  %2043 = ptrtoint ptr %2042 to i64
  %2044 = add i64 %2043, 63
  %2045 = urem i64 %2044, 64
  %2046 = sub i64 %2044, %2045
  %2047 = inttoptr i64 %2046 to ptr
  %2048 = call ptr @_mlir_memref_to_llvm_alloc(i64 448)
  %2049 = ptrtoint ptr %2048 to i64
  %2050 = add i64 %2049, 63
  %2051 = urem i64 %2050, 64
  %2052 = sub i64 %2050, %2051
  %2053 = inttoptr i64 %2052 to ptr
  %2054 = call ptr @_mlir_memref_to_llvm_alloc(i64 96)
  %2055 = ptrtoint ptr %2054 to i64
  %2056 = add i64 %2055, 63
  %2057 = urem i64 %2056, 64
  %2058 = sub i64 %2056, %2057
  %2059 = inttoptr i64 %2058 to ptr
  %2060 = call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %2061 = ptrtoint ptr %2060 to i64
  %2062 = add i64 %2061, 63
  %2063 = urem i64 %2062, 64
  %2064 = sub i64 %2062, %2063
  %2065 = inttoptr i64 %2064 to ptr
  %2066 = call ptr @_mlir_memref_to_llvm_alloc(i64 384)
  %2067 = call ptr @_mlir_memref_to_llvm_alloc(i64 4)
  %2068 = call ptr @_mlir_memref_to_llvm_alloc(i64 4)
  %2069 = call ptr @_mlir_memref_to_llvm_alloc(i64 8)
  %2070 = call ptr @_mlir_memref_to_llvm_alloc(i64 448)
  %2071 = ptrtoint ptr %2070 to i64
  %2072 = add i64 %2071, 63
  %2073 = urem i64 %2072, 64
  %2074 = sub i64 %2072, %2073
  %2075 = inttoptr i64 %2074 to ptr
  %2076 = call ptr @_mlir_memref_to_llvm_alloc(i64 384)
  %2077 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } poison, ptr %2076, 0
  %2078 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %2077, ptr %2076, 1
  %2079 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %2078, i64 0, 2
  %2080 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %2079, i64 4, 3, 0
  %2081 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %2080, i64 8, 3, 1
  %2082 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %2081, i64 3, 3, 2
  %2083 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %2082, i64 24, 4, 0
  %2084 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %2083, i64 3, 4, 1
  %2085 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %2084, i64 1, 4, 2
  %2086 = mul i64 %3, 1
  %2087 = mul i64 %2086, %4
  %2088 = mul i64 %2087, %5
  %2089 = mul i64 %2088, 4
  %2090 = getelementptr inbounds float, ptr %1, i64 %2
  call void @llvm.memcpy.p0.p0.i64(ptr %2076, ptr %2090, i64 %2089, i1 false)
  br label %2091

2091:                                             ; preds = %2441, %2022
  %2092 = phi i64 [ %2463, %2441 ], [ 0, %2022 ]
  %2093 = phi { ptr, ptr, i64, [3 x i64], [3 x i64] } [ %2462, %2441 ], [ %2085, %2022 ]
  %2094 = phi float [ %2446, %2441 ], [ %2034, %2022 ]
  %2095 = phi float [ %2450, %2441 ], [ %2035, %2022 ]
  %2096 = phi double [ %2452, %2441 ], [ 0.000000e+00, %2022 ]
  %2097 = phi double [ %2451, %2441 ], [ 0.000000e+00, %2022 ]
  %2098 = icmp slt i64 %2092, 3
  br i1 %2098, label %2099, label %2464

2099:                                             ; preds = %2091
  store float %2094, ptr %2041, align 4, !tbaa !4
  store float %2095, ptr %2047, align 4, !tbaa !4
  %2100 = call ptr @_mlir_memref_to_llvm_alloc(i64 448)
  %2101 = ptrtoint ptr %2100 to i64
  %2102 = add i64 %2101, 63
  %2103 = urem i64 %2102, 64
  %2104 = sub i64 %2102, %2103
  %2105 = inttoptr i64 %2104 to ptr
  br label %2106

2106:                                             ; preds = %2127, %2099
  %2107 = phi i64 [ %2128, %2127 ], [ 0, %2099 ]
  %2108 = icmp slt i64 %2107, 4
  br i1 %2108, label %2109, label %2129

2109:                                             ; preds = %2106
  br label %2110

2110:                                             ; preds = %2125, %2109
  %2111 = phi i64 [ %2126, %2125 ], [ 0, %2109 ]
  %2112 = icmp slt i64 %2111, 8
  br i1 %2112, label %2113, label %2127

2113:                                             ; preds = %2110
  br label %2114

2114:                                             ; preds = %2117, %2113
  %2115 = phi i64 [ %2124, %2117 ], [ 0, %2113 ]
  %2116 = icmp slt i64 %2115, 3
  br i1 %2116, label %2117, label %2125

2117:                                             ; preds = %2114
  %2118 = load float, ptr @__constant_xf32_2, align 4, !tbaa !4
  %2119 = mul i64 %2107, 24
  %2120 = mul i64 %2111, 3
  %2121 = add i64 %2119, %2120
  %2122 = add i64 %2121, %2115
  %2123 = getelementptr inbounds float, ptr %2053, i64 %2122
  store float %2118, ptr %2123, align 4, !tbaa !4
  %2124 = add i64 %2115, 1
  br label %2114

2125:                                             ; preds = %2114
  %2126 = add i64 %2111, 1
  br label %2110

2127:                                             ; preds = %2110
  %2128 = add i64 %2107, 1
  br label %2106

2129:                                             ; preds = %2106
  %2130 = call ptr @_mlir_memref_to_llvm_alloc(i64 384)
  %2131 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } poison, ptr %2130, 0
  %2132 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %2131, ptr %2130, 1
  %2133 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %2132, i64 0, 2
  %2134 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %2133, i64 4, 3, 0
  %2135 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %2134, i64 8, 3, 1
  %2136 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %2135, i64 3, 3, 2
  %2137 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %2136, i64 24, 4, 0
  %2138 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %2137, i64 3, 4, 1
  %2139 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %2138, i64 1, 4, 2
  call void @llvm.memcpy.p0.p0.i64(ptr %2130, ptr %2053, i64 384, i1 false)
  br label %2140

2140:                                             ; preds = %2258, %2129
  %2141 = phi i64 [ %2281, %2258 ], [ 0, %2129 ]
  %2142 = phi { ptr, ptr, i64, [3 x i64], [3 x i64] } [ %2280, %2258 ], [ %2139, %2129 ]
  %2143 = phi double [ %2261, %2258 ], [ 0.000000e+00, %2129 ]
  %2144 = phi double [ %2263, %2258 ], [ 0.000000e+00, %2129 ]
  %2145 = phi double [ %2264, %2258 ], [ 0.000000e+00, %2129 ]
  %2146 = icmp slt i64 %2141, 32
  br i1 %2146, label %2147, label %2282

2147:                                             ; preds = %2140
  %2148 = mul i64 %2092, 32
  %2149 = add i64 %2033, %2148
  %2150 = add i64 %2149, %2141
  %2151 = srem i64 %2150, 96
  %2152 = icmp ne i64 %2151, 0
  %2153 = icmp slt i64 %2151, 0
  %2154 = and i1 %2153, %2152
  %2155 = add i64 %2151, 96
  %2156 = select i1 %2154, i64 %2155, i64 %2151
  %2157 = icmp slt i64 %2156, 0
  %2158 = add i64 %2156, 96
  %2159 = select i1 %2157, i64 %2158, i64 %2156
  %2160 = icmp sgt i64 %2159, 0
  %2161 = select i1 %2160, i64 %2159, i64 0
  %2162 = icmp slt i64 %2161, 95
  %2163 = select i1 %2162, i64 %2161, i64 95
  %2164 = mul nsw i64 %2163, 8
  %2165 = getelementptr inbounds float, ptr %31, i64 %2163
  %2166 = load float, ptr %2165, align 4, !tbaa !4
  %2167 = getelementptr inbounds float, ptr %36, i64 %2163
  %2168 = load float, ptr %2167, align 4, !tbaa !4
  br label %2169

2169:                                             ; preds = %2172, %2147
  %2170 = phi i64 [ %2177, %2172 ], [ 0, %2147 ]
  %2171 = icmp slt i64 %2170, 8
  br i1 %2171, label %2172, label %2178

2172:                                             ; preds = %2169
  %2173 = getelementptr inbounds float, ptr %24, i64 %2164
  %2174 = getelementptr inbounds float, ptr %2173, i64 %2170
  %2175 = load float, ptr %2174, align 4, !tbaa !4
  %2176 = getelementptr inbounds float, ptr %2059, i64 %2170
  store float %2175, ptr %2176, align 4, !tbaa !4
  %2177 = add i64 %2170, 1
  br label %2169

2178:                                             ; preds = %2169
  %2179 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %2093, 0
  %2180 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %2093, 1
  %2181 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %2093, 2
  %2182 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %2093, 3, 0
  %2183 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %2093, 3, 1
  %2184 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %2093, 3, 2
  %2185 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %2093, 4, 0
  %2186 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %2093, 4, 1
  %2187 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %2093, 4, 2
  %2188 = call { ptr, ptr, i64 } @qnode_forward_0(ptr %2179, ptr %2180, i64 %2181, i64 %2182, i64 %2183, i64 %2184, i64 %2185, i64 %2186, i64 %2187, ptr %2054, ptr %2059, i64 0, i64 8, i64 1)
  %2189 = extractvalue { ptr, ptr, i64 } %2188, 1
  %2190 = load double, ptr %2189, align 8, !tbaa !6
  %2191 = fpext float %2095 to double
  %2192 = fmul double %2191, %2190
  %2193 = fpext float %2094 to double
  %2194 = fadd double %2192, %2193
  %2195 = fpext float %2166 to double
  %2196 = fpext float %2168 to double
  %2197 = fcmp ugt double %2194, 0.000000e+00
  %2198 = select i1 %2197, double %2194, double 0.000000e+00
  %2199 = select i1 false, double 0.000000e+00, double %2198
  %2200 = fcmp une double %2194, %2194
  %2201 = call double @llvm.fabs.f64(double %2194)
  %2202 = fneg double %2201
  %2203 = call double @llvm.exp.f64(double %2202)
  %2204 = fadd double 1.000000e+00, %2203
  %2205 = call double @llvm.log.f64(double %2204)
  %2206 = fadd double %2199, %2205
  %2207 = select i1 %2200, double %2194, double %2206
  %2208 = fmul double %2195, %2194
  %2209 = fsub double %2207, %2208
  %2210 = fmul double %2196, %2209
  store double 0.000000e+00, ptr %2065, align 8, !tbaa !6
  store double 1.000000e+00, ptr %2065, align 8, !tbaa !6
  %2211 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %2093, 0
  %2212 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %2093, 1
  call void @llvm.memset.p0.i64(ptr %2066, i8 0, i64 384, i1 false)
  %2213 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %2093, 2
  %2214 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %2093, 3, 0
  %2215 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %2093, 3, 1
  %2216 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %2093, 3, 2
  %2217 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %2093, 4, 0
  %2218 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %2093, 4, 1
  %2219 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %2093, 4, 2
  call void @llvm.memset.p0.i64(ptr %2067, i8 0, i64 4, i1 false)
  call void @llvm.memset.p0.i64(ptr %2068, i8 0, i64 4, i1 false)
  call void (...) @__enzyme_autodiff0(ptr @_sample_loss.cloned, ptr @enzyme_const, ptr %2211, ptr %2212, ptr %2066, i64 %2213, i64 %2214, i64 %2215, i64 %2216, i64 %2217, i64 %2218, i64 %2219, ptr @enzyme_const, ptr %2036, ptr %2041, ptr %2067, i64 0, ptr @enzyme_const, ptr %2042, ptr %2047, ptr %2068, i64 0, ptr @enzyme_const, ptr %23, ptr @enzyme_const, ptr %24, i64 %2164, i64 8, i64 1, ptr @enzyme_const, ptr %30, ptr @enzyme_const, ptr %31, i64 %2163, ptr @enzyme_const, ptr %35, ptr @enzyme_const, ptr %36, i64 %2163, ptr @enzyme_const, ptr %2069, ptr @enzyme_dupnoneed, ptr %2069, ptr %2065, i64 0)
  %2220 = load float, ptr %2068, align 4, !tbaa !4
  %2221 = load float, ptr %2067, align 4, !tbaa !4
  br label %2222

2222:                                             ; preds = %2256, %2178
  %2223 = phi i64 [ %2257, %2256 ], [ 0, %2178 ]
  %2224 = icmp slt i64 %2223, 4
  br i1 %2224, label %2225, label %2258

2225:                                             ; preds = %2222
  br label %2226

2226:                                             ; preds = %2254, %2225
  %2227 = phi i64 [ %2255, %2254 ], [ 0, %2225 ]
  %2228 = icmp slt i64 %2227, 8
  br i1 %2228, label %2229, label %2256

2229:                                             ; preds = %2226
  br label %2230

2230:                                             ; preds = %2233, %2229
  %2231 = phi i64 [ %2253, %2233 ], [ 0, %2229 ]
  %2232 = icmp slt i64 %2231, 3
  br i1 %2232, label %2233, label %2254

2233:                                             ; preds = %2230
  %2234 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %2142, 1
  %2235 = mul i64 %2223, 24
  %2236 = mul i64 %2227, 3
  %2237 = add i64 %2235, %2236
  %2238 = add i64 %2237, %2231
  %2239 = getelementptr inbounds float, ptr %2234, i64 %2238
  %2240 = load float, ptr %2239, align 4, !tbaa !4
  %2241 = mul i64 %2223, 24
  %2242 = mul i64 %2227, 3
  %2243 = add i64 %2241, %2242
  %2244 = add i64 %2243, %2231
  %2245 = getelementptr inbounds float, ptr %2066, i64 %2244
  %2246 = load float, ptr %2245, align 4, !tbaa !4
  %2247 = fadd float %2240, %2246
  %2248 = mul i64 %2223, 24
  %2249 = mul i64 %2227, 3
  %2250 = add i64 %2248, %2249
  %2251 = add i64 %2250, %2231
  %2252 = getelementptr inbounds float, ptr %2105, i64 %2251
  store float %2247, ptr %2252, align 4, !tbaa !4
  %2253 = add i64 %2231, 1
  br label %2230

2254:                                             ; preds = %2230
  %2255 = add i64 %2227, 1
  br label %2226

2256:                                             ; preds = %2226
  %2257 = add i64 %2223, 1
  br label %2222

2258:                                             ; preds = %2222
  %2259 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %2142, 0
  call void @_mlir_memref_to_llvm_free(ptr %2259)
  %2260 = fpext float %2221 to double
  %2261 = fadd double %2143, %2260
  %2262 = fpext float %2220 to double
  %2263 = fadd double %2144, %2262
  %2264 = fadd double %2145, %2210
  %2265 = call ptr @_mlir_memref_to_llvm_alloc(i64 448)
  %2266 = ptrtoint ptr %2265 to i64
  %2267 = add i64 %2266, 63
  %2268 = urem i64 %2267, 64
  %2269 = sub i64 %2267, %2268
  %2270 = inttoptr i64 %2269 to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr %2270, ptr %2105, i64 384, i1 false)
  %2271 = call ptr @_mlir_memref_to_llvm_alloc(i64 384)
  %2272 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } poison, ptr %2271, 0
  %2273 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %2272, ptr %2271, 1
  %2274 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %2273, i64 0, 2
  %2275 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %2274, i64 4, 3, 0
  %2276 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %2275, i64 8, 3, 1
  %2277 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %2276, i64 3, 3, 2
  %2278 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %2277, i64 24, 4, 0
  %2279 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %2278, i64 3, 4, 1
  %2280 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %2279, i64 1, 4, 2
  call void @llvm.memcpy.p0.p0.i64(ptr %2271, ptr %2270, i64 384, i1 false)
  call void @_mlir_memref_to_llvm_free(ptr %2265)
  %2281 = add i64 %2141, 1
  br label %2140

2282:                                             ; preds = %2140
  br label %2283

2283:                                             ; preds = %2304, %2282
  %2284 = phi i64 [ %2305, %2304 ], [ 0, %2282 ]
  %2285 = icmp slt i64 %2284, 4
  br i1 %2285, label %2286, label %2306

2286:                                             ; preds = %2283
  br label %2287

2287:                                             ; preds = %2302, %2286
  %2288 = phi i64 [ %2303, %2302 ], [ 0, %2286 ]
  %2289 = icmp slt i64 %2288, 8
  br i1 %2289, label %2290, label %2304

2290:                                             ; preds = %2287
  br label %2291

2291:                                             ; preds = %2294, %2290
  %2292 = phi i64 [ %2301, %2294 ], [ 0, %2290 ]
  %2293 = icmp slt i64 %2292, 3
  br i1 %2293, label %2294, label %2302

2294:                                             ; preds = %2291
  %2295 = load float, ptr @__constant_xf32_0, align 4, !tbaa !4
  %2296 = mul i64 %2284, 24
  %2297 = mul i64 %2288, 3
  %2298 = add i64 %2296, %2297
  %2299 = add i64 %2298, %2292
  %2300 = getelementptr inbounds float, ptr %2105, i64 %2299
  store float %2295, ptr %2300, align 4, !tbaa !4
  %2301 = add i64 %2292, 1
  br label %2291

2302:                                             ; preds = %2291
  %2303 = add i64 %2288, 1
  br label %2287

2304:                                             ; preds = %2287
  %2305 = add i64 %2284, 1
  br label %2283

2306:                                             ; preds = %2283
  br label %2307

2307:                                             ; preds = %2341, %2306
  %2308 = phi i64 [ %2342, %2341 ], [ 0, %2306 ]
  %2309 = icmp slt i64 %2308, 4
  br i1 %2309, label %2310, label %2343

2310:                                             ; preds = %2307
  br label %2311

2311:                                             ; preds = %2339, %2310
  %2312 = phi i64 [ %2340, %2339 ], [ 0, %2310 ]
  %2313 = icmp slt i64 %2312, 8
  br i1 %2313, label %2314, label %2341

2314:                                             ; preds = %2311
  br label %2315

2315:                                             ; preds = %2318, %2314
  %2316 = phi i64 [ %2338, %2318 ], [ 0, %2314 ]
  %2317 = icmp slt i64 %2316, 3
  br i1 %2317, label %2318, label %2339

2318:                                             ; preds = %2315
  %2319 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %2142, 1
  %2320 = mul i64 %2308, 24
  %2321 = mul i64 %2312, 3
  %2322 = add i64 %2320, %2321
  %2323 = add i64 %2322, %2316
  %2324 = getelementptr inbounds float, ptr %2319, i64 %2323
  %2325 = load float, ptr %2324, align 4, !tbaa !4
  %2326 = mul i64 %2308, 24
  %2327 = mul i64 %2312, 3
  %2328 = add i64 %2326, %2327
  %2329 = add i64 %2328, %2316
  %2330 = getelementptr inbounds float, ptr %2105, i64 %2329
  %2331 = load float, ptr %2330, align 4, !tbaa !4
  %2332 = fmul float %2325, %2331
  %2333 = mul i64 %2308, 24
  %2334 = mul i64 %2312, 3
  %2335 = add i64 %2333, %2334
  %2336 = add i64 %2335, %2316
  %2337 = getelementptr inbounds float, ptr %2075, i64 %2336
  store float %2332, ptr %2337, align 4, !tbaa !4
  %2338 = add i64 %2316, 1
  br label %2315

2339:                                             ; preds = %2315
  %2340 = add i64 %2312, 1
  br label %2311

2341:                                             ; preds = %2311
  %2342 = add i64 %2308, 1
  br label %2307

2343:                                             ; preds = %2307
  %2344 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %2142, 0
  call void @_mlir_memref_to_llvm_free(ptr %2344)
  br label %2345

2345:                                             ; preds = %2366, %2343
  %2346 = phi i64 [ %2367, %2366 ], [ 0, %2343 ]
  %2347 = icmp slt i64 %2346, 4
  br i1 %2347, label %2348, label %2368

2348:                                             ; preds = %2345
  br label %2349

2349:                                             ; preds = %2364, %2348
  %2350 = phi i64 [ %2365, %2364 ], [ 0, %2348 ]
  %2351 = icmp slt i64 %2350, 8
  br i1 %2351, label %2352, label %2366

2352:                                             ; preds = %2349
  br label %2353

2353:                                             ; preds = %2356, %2352
  %2354 = phi i64 [ %2363, %2356 ], [ 0, %2352 ]
  %2355 = icmp slt i64 %2354, 3
  br i1 %2355, label %2356, label %2364

2356:                                             ; preds = %2353
  %2357 = load float, ptr @__constant_xf32_1, align 4, !tbaa !4
  %2358 = mul i64 %2346, 24
  %2359 = mul i64 %2350, 3
  %2360 = add i64 %2358, %2359
  %2361 = add i64 %2360, %2354
  %2362 = getelementptr inbounds float, ptr %2105, i64 %2361
  store float %2357, ptr %2362, align 4, !tbaa !4
  %2363 = add i64 %2354, 1
  br label %2353

2364:                                             ; preds = %2353
  %2365 = add i64 %2350, 1
  br label %2349

2366:                                             ; preds = %2349
  %2367 = add i64 %2346, 1
  br label %2345

2368:                                             ; preds = %2345
  br label %2369

2369:                                             ; preds = %2402, %2368
  %2370 = phi i64 [ %2403, %2402 ], [ 0, %2368 ]
  %2371 = icmp slt i64 %2370, 4
  br i1 %2371, label %2372, label %2404

2372:                                             ; preds = %2369
  br label %2373

2373:                                             ; preds = %2400, %2372
  %2374 = phi i64 [ %2401, %2400 ], [ 0, %2372 ]
  %2375 = icmp slt i64 %2374, 8
  br i1 %2375, label %2376, label %2402

2376:                                             ; preds = %2373
  br label %2377

2377:                                             ; preds = %2380, %2376
  %2378 = phi i64 [ %2399, %2380 ], [ 0, %2376 ]
  %2379 = icmp slt i64 %2378, 3
  br i1 %2379, label %2380, label %2400

2380:                                             ; preds = %2377
  %2381 = mul i64 %2370, 24
  %2382 = mul i64 %2374, 3
  %2383 = add i64 %2381, %2382
  %2384 = add i64 %2383, %2378
  %2385 = getelementptr inbounds float, ptr %2105, i64 %2384
  %2386 = load float, ptr %2385, align 4, !tbaa !4
  %2387 = mul i64 %2370, 24
  %2388 = mul i64 %2374, 3
  %2389 = add i64 %2387, %2388
  %2390 = add i64 %2389, %2378
  %2391 = getelementptr inbounds float, ptr %2075, i64 %2390
  %2392 = load float, ptr %2391, align 4, !tbaa !4
  %2393 = fmul float %2386, %2392
  %2394 = mul i64 %2370, 24
  %2395 = mul i64 %2374, 3
  %2396 = add i64 %2394, %2395
  %2397 = add i64 %2396, %2378
  %2398 = getelementptr inbounds float, ptr %2105, i64 %2397
  store float %2393, ptr %2398, align 4, !tbaa !4
  %2399 = add i64 %2378, 1
  br label %2377

2400:                                             ; preds = %2377
  %2401 = add i64 %2374, 1
  br label %2373

2402:                                             ; preds = %2373
  %2403 = add i64 %2370, 1
  br label %2369

2404:                                             ; preds = %2369
  br label %2405

2405:                                             ; preds = %2439, %2404
  %2406 = phi i64 [ %2440, %2439 ], [ 0, %2404 ]
  %2407 = icmp slt i64 %2406, 4
  br i1 %2407, label %2408, label %2441

2408:                                             ; preds = %2405
  br label %2409

2409:                                             ; preds = %2437, %2408
  %2410 = phi i64 [ %2438, %2437 ], [ 0, %2408 ]
  %2411 = icmp slt i64 %2410, 8
  br i1 %2411, label %2412, label %2439

2412:                                             ; preds = %2409
  br label %2413

2413:                                             ; preds = %2416, %2412
  %2414 = phi i64 [ %2436, %2416 ], [ 0, %2412 ]
  %2415 = icmp slt i64 %2414, 3
  br i1 %2415, label %2416, label %2437

2416:                                             ; preds = %2413
  %2417 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %2093, 1
  %2418 = mul i64 %2406, 24
  %2419 = mul i64 %2410, 3
  %2420 = add i64 %2418, %2419
  %2421 = add i64 %2420, %2414
  %2422 = getelementptr inbounds float, ptr %2417, i64 %2421
  %2423 = load float, ptr %2422, align 4, !tbaa !4
  %2424 = mul i64 %2406, 24
  %2425 = mul i64 %2410, 3
  %2426 = add i64 %2424, %2425
  %2427 = add i64 %2426, %2414
  %2428 = getelementptr inbounds float, ptr %2105, i64 %2427
  %2429 = load float, ptr %2428, align 4, !tbaa !4
  %2430 = fsub float %2423, %2429
  %2431 = mul i64 %2406, 24
  %2432 = mul i64 %2410, 3
  %2433 = add i64 %2431, %2432
  %2434 = add i64 %2433, %2414
  %2435 = getelementptr inbounds float, ptr %2105, i64 %2434
  store float %2430, ptr %2435, align 4, !tbaa !4
  %2436 = add i64 %2414, 1
  br label %2413

2437:                                             ; preds = %2413
  %2438 = add i64 %2410, 1
  br label %2409

2439:                                             ; preds = %2409
  %2440 = add i64 %2406, 1
  br label %2405

2441:                                             ; preds = %2405
  %2442 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %2093, 0
  call void @_mlir_memref_to_llvm_free(ptr %2442)
  %2443 = fmul double %2143, 3.125000e-02
  %2444 = fptrunc double %2443 to float
  %2445 = fmul float %2444, 0x3F847AE140000000
  %2446 = fsub float %2094, %2445
  %2447 = fmul double %2144, 3.125000e-02
  %2448 = fptrunc double %2447 to float
  %2449 = fmul float %2448, 0x3F847AE140000000
  %2450 = fsub float %2095, %2449
  %2451 = fdiv double %2145, 3.200000e+01
  %2452 = fadd double %2096, %2451
  %2453 = call ptr @_mlir_memref_to_llvm_alloc(i64 384)
  %2454 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } poison, ptr %2453, 0
  %2455 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %2454, ptr %2453, 1
  %2456 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %2455, i64 0, 2
  %2457 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %2456, i64 4, 3, 0
  %2458 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %2457, i64 8, 3, 1
  %2459 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %2458, i64 3, 3, 2
  %2460 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %2459, i64 24, 4, 0
  %2461 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %2460, i64 3, 4, 1
  %2462 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %2461, i64 1, 4, 2
  call void @llvm.memcpy.p0.p0.i64(ptr %2453, ptr %2105, i64 384, i1 false)
  call void @_mlir_memref_to_llvm_free(ptr %2100)
  %2463 = add i64 %2092, 1
  br label %2091

2464:                                             ; preds = %2091
  call void @_mlir_memref_to_llvm_free(ptr %2070)
  call void @_mlir_memref_to_llvm_free(ptr %2069)
  call void @_mlir_memref_to_llvm_free(ptr %2068)
  call void @_mlir_memref_to_llvm_free(ptr %2067)
  call void @_mlir_memref_to_llvm_free(ptr %2066)
  call void @_mlir_memref_to_llvm_free(ptr %2060)
  call void @_mlir_memref_to_llvm_free(ptr %2054)
  call void @_mlir_memref_to_llvm_free(ptr %2048)
  call void @_mlir_memref_to_llvm_free(ptr %2042)
  call void @_mlir_memref_to_llvm_free(ptr %2036)
  %2465 = call ptr @_mlir_memref_to_llvm_alloc(i64 68)
  %2466 = ptrtoint ptr %2465 to i64
  %2467 = add i64 %2466, 63
  %2468 = urem i64 %2467, 64
  %2469 = sub i64 %2467, %2468
  %2470 = inttoptr i64 %2469 to ptr
  %2471 = insertvalue { ptr, ptr, i64 } poison, ptr %2465, 0
  %2472 = insertvalue { ptr, ptr, i64 } %2471, ptr %2470, 1
  %2473 = insertvalue { ptr, ptr, i64 } %2472, i64 0, 2
  store float %2094, ptr %2470, align 4, !tbaa !4
  %2474 = call ptr @_mlir_memref_to_llvm_alloc(i64 68)
  %2475 = ptrtoint ptr %2474 to i64
  %2476 = add i64 %2475, 63
  %2477 = urem i64 %2476, 64
  %2478 = sub i64 %2476, %2477
  %2479 = inttoptr i64 %2478 to ptr
  %2480 = insertvalue { ptr, ptr, i64 } poison, ptr %2474, 0
  %2481 = insertvalue { ptr, ptr, i64 } %2480, ptr %2479, 1
  %2482 = insertvalue { ptr, ptr, i64 } %2481, i64 0, 2
  store float %2095, ptr %2479, align 4, !tbaa !4
  %2483 = fdiv double %2096, 3.000000e+00
  %2484 = call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %2485 = ptrtoint ptr %2484 to i64
  %2486 = add i64 %2485, 63
  %2487 = urem i64 %2486, 64
  %2488 = sub i64 %2486, %2487
  %2489 = inttoptr i64 %2488 to ptr
  store double %2483, ptr %2489, align 8, !tbaa !6
  %2490 = call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %2491 = ptrtoint ptr %2490 to i64
  %2492 = add i64 %2491, 63
  %2493 = urem i64 %2492, 64
  %2494 = sub i64 %2492, %2493
  %2495 = inttoptr i64 %2494 to ptr
  store double %2097, ptr %2495, align 8, !tbaa !6
  %2496 = call ptr @_mlir_memref_to_llvm_alloc(i64 80)
  %2497 = ptrtoint ptr %2496 to i64
  %2498 = add i64 %2497, 63
  %2499 = urem i64 %2498, 64
  %2500 = sub i64 %2498, %2499
  %2501 = inttoptr i64 %2500 to ptr
  %2502 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %2496, 0
  %2503 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %2502, ptr %2501, 1
  %2504 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %2503, i64 0, 2
  %2505 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %2504, i64 2, 3, 0
  %2506 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %2505, i64 1, 4, 0
  br label %2507

2507:                                             ; preds = %2521, %2464
  %2508 = phi i64 [ %2523, %2521 ], [ 0, %2464 ]
  %2509 = icmp slt i64 %2508, 2
  br i1 %2509, label %2510, label %2524

2510:                                             ; preds = %2507
  %2511 = icmp ult i64 %2508, 1
  br i1 %2511, label %2512, label %2515

2512:                                             ; preds = %2510
  %2513 = getelementptr inbounds double, ptr %2489, i64 %2508
  %2514 = load double, ptr %2513, align 8, !tbaa !6
  br label %2519

2515:                                             ; preds = %2510
  %2516 = sub i64 %2508, 1
  %2517 = getelementptr inbounds double, ptr %2495, i64 %2516
  %2518 = load double, ptr %2517, align 8, !tbaa !6
  br label %2519

2519:                                             ; preds = %2512, %2515
  %2520 = phi double [ %2518, %2515 ], [ %2514, %2512 ]
  br label %2521

2521:                                             ; preds = %2519
  %2522 = getelementptr inbounds double, ptr %2501, i64 %2508
  store double %2520, ptr %2522, align 8, !tbaa !6
  %2523 = add i64 %2508, 1
  br label %2507

2524:                                             ; preds = %2507
  call void @_mlir_memref_to_llvm_free(ptr %2490)
  call void @_mlir_memref_to_llvm_free(ptr %2484)
  %2525 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %2093, 0
  %2526 = ptrtoint ptr %2525 to i64
  %2527 = icmp eq i64 3735928559, %2526
  br i1 %2527, label %2528, label %2549

2528:                                             ; preds = %2524
  %2529 = call ptr @_mlir_memref_to_llvm_alloc(i64 384)
  %2530 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } poison, ptr %2529, 0
  %2531 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %2530, ptr %2529, 1
  %2532 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %2531, i64 0, 2
  %2533 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %2532, i64 4, 3, 0
  %2534 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %2533, i64 8, 3, 1
  %2535 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %2534, i64 3, 3, 2
  %2536 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %2535, i64 24, 4, 0
  %2537 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %2536, i64 3, 4, 1
  %2538 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %2537, i64 1, 4, 2
  %2539 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %2093, 3, 0
  %2540 = mul i64 %2539, 1
  %2541 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %2093, 3, 1
  %2542 = mul i64 %2540, %2541
  %2543 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %2093, 3, 2
  %2544 = mul i64 %2542, %2543
  %2545 = mul i64 %2544, 4
  %2546 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %2093, 1
  %2547 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %2093, 2
  %2548 = getelementptr inbounds float, ptr %2546, i64 %2547
  call void @llvm.memcpy.p0.p0.i64(ptr %2529, ptr %2548, i64 %2545, i1 false)
  br label %2550

2549:                                             ; preds = %2524
  br label %2550

2550:                                             ; preds = %2528, %2549
  %2551 = phi { ptr, ptr, i64, [3 x i64], [3 x i64] } [ %2093, %2549 ], [ %2538, %2528 ]
  br label %2552

2552:                                             ; preds = %2550
  %2553 = ptrtoint ptr %2465 to i64
  %2554 = icmp eq i64 3735928559, %2553
  br i1 %2554, label %2555, label %2560

2555:                                             ; preds = %2552
  %2556 = call ptr @_mlir_memref_to_llvm_alloc(i64 4)
  %2557 = insertvalue { ptr, ptr, i64 } poison, ptr %2556, 0
  %2558 = insertvalue { ptr, ptr, i64 } %2557, ptr %2556, 1
  %2559 = insertvalue { ptr, ptr, i64 } %2558, i64 0, 2
  call void @llvm.memcpy.p0.p0.i64(ptr %2556, ptr %2470, i64 4, i1 false)
  br label %2561

2560:                                             ; preds = %2552
  br label %2561

2561:                                             ; preds = %2555, %2560
  %2562 = phi { ptr, ptr, i64 } [ %2473, %2560 ], [ %2559, %2555 ]
  br label %2563

2563:                                             ; preds = %2561
  %2564 = ptrtoint ptr %2474 to i64
  %2565 = icmp eq i64 3735928559, %2564
  br i1 %2565, label %2566, label %2571

2566:                                             ; preds = %2563
  %2567 = call ptr @_mlir_memref_to_llvm_alloc(i64 4)
  %2568 = insertvalue { ptr, ptr, i64 } poison, ptr %2567, 0
  %2569 = insertvalue { ptr, ptr, i64 } %2568, ptr %2567, 1
  %2570 = insertvalue { ptr, ptr, i64 } %2569, i64 0, 2
  call void @llvm.memcpy.p0.p0.i64(ptr %2567, ptr %2479, i64 4, i1 false)
  br label %2572

2571:                                             ; preds = %2563
  br label %2572

2572:                                             ; preds = %2566, %2571
  %2573 = phi { ptr, ptr, i64 } [ %2482, %2571 ], [ %2570, %2566 ]
  br label %2574

2574:                                             ; preds = %2572
  %2575 = ptrtoint ptr %15 to i64
  %2576 = icmp eq i64 3735928559, %2575
  br i1 %2576, label %2577, label %2583

2577:                                             ; preds = %2574
  %2578 = call ptr @_mlir_memref_to_llvm_alloc(i64 4)
  %2579 = insertvalue { ptr, ptr, i64 } poison, ptr %2578, 0
  %2580 = insertvalue { ptr, ptr, i64 } %2579, ptr %2578, 1
  %2581 = insertvalue { ptr, ptr, i64 } %2580, i64 0, 2
  %2582 = getelementptr inbounds i32, ptr %16, i64 %17
  call void @llvm.memcpy.p0.p0.i64(ptr %2578, ptr %2582, i64 4, i1 false)
  br label %2584

2583:                                             ; preds = %2574
  br label %2584

2584:                                             ; preds = %2577, %2583
  %2585 = phi { ptr, ptr, i64 } [ %43, %2583 ], [ %2581, %2577 ]
  br label %2586

2586:                                             ; preds = %2584
  %2587 = ptrtoint ptr %846 to i64
  %2588 = icmp eq i64 3735928559, %2587
  br i1 %2588, label %2589, label %2596

2589:                                             ; preds = %2586
  %2590 = call ptr @_mlir_memref_to_llvm_alloc(i64 8)
  %2591 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %2590, 0
  %2592 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %2591, ptr %2590, 1
  %2593 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %2592, i64 0, 2
  %2594 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %2593, i64 2, 3, 0
  %2595 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %2594, i64 1, 4, 0
  call void @llvm.memcpy.p0.p0.i64(ptr %2590, ptr %851, i64 8, i1 false)
  br label %2597

2596:                                             ; preds = %2586
  br label %2597

2597:                                             ; preds = %2589, %2596
  %2598 = phi { ptr, ptr, i64, [1 x i64], [1 x i64] } [ %886, %2596 ], [ %2595, %2589 ]
  br label %2599

2599:                                             ; preds = %2597
  %2600 = ptrtoint ptr %2496 to i64
  %2601 = icmp eq i64 3735928559, %2600
  br i1 %2601, label %2602, label %2609

2602:                                             ; preds = %2599
  %2603 = call ptr @_mlir_memref_to_llvm_alloc(i64 16)
  %2604 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %2603, 0
  %2605 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %2604, ptr %2603, 1
  %2606 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %2605, i64 0, 2
  %2607 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %2606, i64 2, 3, 0
  %2608 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %2607, i64 1, 4, 0
  call void @llvm.memcpy.p0.p0.i64(ptr %2603, ptr %2501, i64 16, i1 false)
  br label %2610

2609:                                             ; preds = %2599
  br label %2610

2610:                                             ; preds = %2602, %2609
  %2611 = phi { ptr, ptr, i64, [1 x i64], [1 x i64] } [ %2506, %2609 ], [ %2608, %2602 ]
  br label %2612

2612:                                             ; preds = %2610
  %2613 = insertvalue { { ptr, ptr, i64, [3 x i64], [3 x i64] }, { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64, [1 x i64], [1 x i64] }, { ptr, ptr, i64, [1 x i64], [1 x i64] } } poison, { ptr, ptr, i64, [3 x i64], [3 x i64] } %2551, 0
  %2614 = insertvalue { { ptr, ptr, i64, [3 x i64], [3 x i64] }, { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64, [1 x i64], [1 x i64] }, { ptr, ptr, i64, [1 x i64], [1 x i64] } } %2613, { ptr, ptr, i64 } %2562, 1
  %2615 = insertvalue { { ptr, ptr, i64, [3 x i64], [3 x i64] }, { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64, [1 x i64], [1 x i64] }, { ptr, ptr, i64, [1 x i64], [1 x i64] } } %2614, { ptr, ptr, i64 } %2573, 2
  %2616 = insertvalue { { ptr, ptr, i64, [3 x i64], [3 x i64] }, { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64, [1 x i64], [1 x i64] }, { ptr, ptr, i64, [1 x i64], [1 x i64] } } %2615, { ptr, ptr, i64 } %2585, 3
  %2617 = insertvalue { { ptr, ptr, i64, [3 x i64], [3 x i64] }, { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64, [1 x i64], [1 x i64] }, { ptr, ptr, i64, [1 x i64], [1 x i64] } } %2616, { ptr, ptr, i64, [1 x i64], [1 x i64] } %2598, 4
  %2618 = insertvalue { { ptr, ptr, i64, [3 x i64], [3 x i64] }, { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64, [1 x i64], [1 x i64] }, { ptr, ptr, i64, [1 x i64], [1 x i64] } } %2617, { ptr, ptr, i64, [1 x i64], [1 x i64] } %2611, 5
  ret { { ptr, ptr, i64, [3 x i64], [3 x i64] }, { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64, [1 x i64], [1 x i64] }, { ptr, ptr, i64, [1 x i64], [1 x i64] } } %2618
}

define void @_catalyst_pyface_jit_train_epoch_compiled(ptr %0, ptr %1) {
  %3 = load { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %1, align 8
  %4 = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %3, 0
  %5 = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %3, 1
  %6 = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %3, 2
  %7 = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %3, 3
  %8 = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %3, 4
  %9 = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %3, 5
  %10 = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %3, 6
  %11 = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %3, 7
  call void @_catalyst_ciface_jit_train_epoch_compiled(ptr %0, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, ptr %9, ptr %10, ptr %11)
  ret void
}

define void @_catalyst_ciface_jit_train_epoch_compiled(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8) {
  %10 = load { ptr, ptr, i64, [3 x i64], [3 x i64] }, ptr %1, align 8
  %11 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %10, 0
  %12 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %10, 1
  %13 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %10, 2
  %14 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %10, 3, 0
  %15 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %10, 3, 1
  %16 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %10, 3, 2
  %17 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %10, 4, 0
  %18 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %10, 4, 1
  %19 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %10, 4, 2
  %20 = load { ptr, ptr, i64 }, ptr %2, align 8
  %21 = extractvalue { ptr, ptr, i64 } %20, 0
  %22 = extractvalue { ptr, ptr, i64 } %20, 1
  %23 = extractvalue { ptr, ptr, i64 } %20, 2
  %24 = load { ptr, ptr, i64 }, ptr %3, align 8
  %25 = extractvalue { ptr, ptr, i64 } %24, 0
  %26 = extractvalue { ptr, ptr, i64 } %24, 1
  %27 = extractvalue { ptr, ptr, i64 } %24, 2
  %28 = load { ptr, ptr, i64 }, ptr %4, align 8
  %29 = extractvalue { ptr, ptr, i64 } %28, 0
  %30 = extractvalue { ptr, ptr, i64 } %28, 1
  %31 = extractvalue { ptr, ptr, i64 } %28, 2
  %32 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %5, align 8
  %33 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %32, 0
  %34 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %32, 1
  %35 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %32, 2
  %36 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %32, 3, 0
  %37 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %32, 4, 0
  %38 = load { ptr, ptr, i64, [2 x i64], [2 x i64] }, ptr %6, align 8
  %39 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %38, 0
  %40 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %38, 1
  %41 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %38, 2
  %42 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %38, 3, 0
  %43 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %38, 3, 1
  %44 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %38, 4, 0
  %45 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %38, 4, 1
  %46 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %7, align 8
  %47 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %46, 0
  %48 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %46, 1
  %49 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %46, 2
  %50 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %46, 3, 0
  %51 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %46, 4, 0
  %52 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %8, align 8
  %53 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %52, 0
  %54 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %52, 1
  %55 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %52, 2
  %56 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %52, 3, 0
  %57 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %52, 4, 0
  %58 = call { { ptr, ptr, i64, [3 x i64], [3 x i64] }, { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64, [1 x i64], [1 x i64] }, { ptr, ptr, i64, [1 x i64], [1 x i64] } } @jit_train_epoch_compiled(ptr %11, ptr %12, i64 %13, i64 %14, i64 %15, i64 %16, i64 %17, i64 %18, i64 %19, ptr %21, ptr %22, i64 %23, ptr %25, ptr %26, i64 %27, ptr %29, ptr %30, i64 %31, ptr %33, ptr %34, i64 %35, i64 %36, i64 %37, ptr %39, ptr %40, i64 %41, i64 %42, i64 %43, i64 %44, i64 %45, ptr %47, ptr %48, i64 %49, i64 %50, i64 %51, ptr %53, ptr %54, i64 %55, i64 %56, i64 %57)
  store { { ptr, ptr, i64, [3 x i64], [3 x i64] }, { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64, [1 x i64], [1 x i64] }, { ptr, ptr, i64, [1 x i64], [1 x i64] } } %58, ptr %0, align 8
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
  call void @__catalyst__rt__device_init(ptr @"/home/pichau/QAgents/.venv/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.so", ptr @LightningSimulator, ptr @"{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}", i64 0, i1 false)
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
  store double %369, ptr %375, align 8, !tbaa !6
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
  call void @__catalyst__rt__device_init(ptr @"/home/pichau/QAgents/.venv/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.so", ptr @LightningSimulator, ptr @"{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}", i64 0, i1 false)
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
  store i64 0, ptr %15, align 4, !tbaa !1
  %16 = load i64, ptr %15, align 4, !tbaa !1
  %17 = add i64 %16, 1
  store i64 %17, ptr %15, align 4, !tbaa !1
  %18 = load i64, ptr %15, align 4, !tbaa !1
  %19 = add i64 %18, 1
  store i64 %19, ptr %15, align 4, !tbaa !1
  %20 = load i64, ptr %15, align 4, !tbaa !1
  %21 = add i64 %20, 1
  store i64 %21, ptr %15, align 4, !tbaa !1
  %22 = load i64, ptr %15, align 4, !tbaa !1
  %23 = add i64 %22, 1
  store i64 %23, ptr %15, align 4, !tbaa !1
  %24 = load i64, ptr %15, align 4, !tbaa !1
  %25 = add i64 %24, 1
  store i64 %25, ptr %15, align 4, !tbaa !1
  %26 = load i64, ptr %15, align 4, !tbaa !1
  %27 = add i64 %26, 1
  store i64 %27, ptr %15, align 4, !tbaa !1
  %28 = load i64, ptr %15, align 4, !tbaa !1
  %29 = add i64 %28, 1
  store i64 %29, ptr %15, align 4, !tbaa !1
  %30 = load i64, ptr %15, align 4, !tbaa !1
  %31 = add i64 %30, 1
  store i64 %31, ptr %15, align 4, !tbaa !1
  %32 = load i64, ptr %15, align 4, !tbaa !1
  %33 = add i64 %32, 1
  store i64 %33, ptr %15, align 4, !tbaa !1
  %34 = load i64, ptr %15, align 4, !tbaa !1
  %35 = add i64 %34, 1
  store i64 %35, ptr %15, align 4, !tbaa !1
  %36 = load i64, ptr %15, align 4, !tbaa !1
  %37 = add i64 %36, 1
  store i64 %37, ptr %15, align 4, !tbaa !1
  %38 = load i64, ptr %15, align 4, !tbaa !1
  %39 = add i64 %38, 1
  store i64 %39, ptr %15, align 4, !tbaa !1
  %40 = load i64, ptr %15, align 4, !tbaa !1
  %41 = add i64 %40, 1
  store i64 %41, ptr %15, align 4, !tbaa !1
  %42 = load i64, ptr %15, align 4, !tbaa !1
  %43 = add i64 %42, 1
  store i64 %43, ptr %15, align 4, !tbaa !1
  %44 = load i64, ptr %15, align 4, !tbaa !1
  %45 = add i64 %44, 1
  store i64 %45, ptr %15, align 4, !tbaa !1
  %46 = load i64, ptr %15, align 4, !tbaa !1
  %47 = add i64 %46, 1
  store i64 %47, ptr %15, align 4, !tbaa !1
  %48 = load i64, ptr %15, align 4, !tbaa !1
  %49 = add i64 %48, 1
  store i64 %49, ptr %15, align 4, !tbaa !1
  %50 = load i64, ptr %15, align 4, !tbaa !1
  %51 = add i64 %50, 1
  store i64 %51, ptr %15, align 4, !tbaa !1
  %52 = load i64, ptr %15, align 4, !tbaa !1
  %53 = add i64 %52, 1
  store i64 %53, ptr %15, align 4, !tbaa !1
  %54 = load i64, ptr %15, align 4, !tbaa !1
  %55 = add i64 %54, 1
  store i64 %55, ptr %15, align 4, !tbaa !1
  %56 = load i64, ptr %15, align 4, !tbaa !1
  %57 = add i64 %56, 1
  store i64 %57, ptr %15, align 4, !tbaa !1
  %58 = load i64, ptr %15, align 4, !tbaa !1
  %59 = add i64 %58, 1
  store i64 %59, ptr %15, align 4, !tbaa !1
  %60 = load i64, ptr %15, align 4, !tbaa !1
  %61 = add i64 %60, 1
  store i64 %61, ptr %15, align 4, !tbaa !1
  %62 = load i64, ptr %15, align 4, !tbaa !1
  %63 = add i64 %62, 1
  store i64 %63, ptr %15, align 4, !tbaa !1
  %64 = load i64, ptr %15, align 4, !tbaa !1
  %65 = add i64 %64, 1
  store i64 %65, ptr %15, align 4, !tbaa !1
  %66 = load i64, ptr %15, align 4, !tbaa !1
  %67 = add i64 %66, 1
  store i64 %67, ptr %15, align 4, !tbaa !1
  %68 = load i64, ptr %15, align 4, !tbaa !1
  %69 = add i64 %68, 1
  store i64 %69, ptr %15, align 4, !tbaa !1
  %70 = load i64, ptr %15, align 4, !tbaa !1
  %71 = add i64 %70, 1
  store i64 %71, ptr %15, align 4, !tbaa !1
  %72 = load i64, ptr %15, align 4, !tbaa !1
  %73 = add i64 %72, 1
  store i64 %73, ptr %15, align 4, !tbaa !1
  %74 = load i64, ptr %15, align 4, !tbaa !1
  %75 = add i64 %74, 1
  store i64 %75, ptr %15, align 4, !tbaa !1
  %76 = load i64, ptr %15, align 4, !tbaa !1
  %77 = add i64 %76, 1
  store i64 %77, ptr %15, align 4, !tbaa !1
  %78 = load i64, ptr %15, align 4, !tbaa !1
  %79 = add i64 %78, 1
  store i64 %79, ptr %15, align 4, !tbaa !1
  %80 = load i64, ptr %15, align 4, !tbaa !1
  %81 = add i64 %80, 1
  store i64 %81, ptr %15, align 4, !tbaa !1
  %82 = load i64, ptr %15, align 4, !tbaa !1
  %83 = add i64 %82, 1
  store i64 %83, ptr %15, align 4, !tbaa !1
  %84 = load i64, ptr %15, align 4, !tbaa !1
  %85 = add i64 %84, 1
  store i64 %85, ptr %15, align 4, !tbaa !1
  %86 = load i64, ptr %15, align 4, !tbaa !1
  %87 = add i64 %86, 1
  store i64 %87, ptr %15, align 4, !tbaa !1
  %88 = load i64, ptr %15, align 4, !tbaa !1
  %89 = add i64 %88, 1
  store i64 %89, ptr %15, align 4, !tbaa !1
  %90 = load i64, ptr %15, align 4, !tbaa !1
  %91 = add i64 %90, 1
  store i64 %91, ptr %15, align 4, !tbaa !1
  %92 = load i64, ptr %15, align 4, !tbaa !1
  %93 = add i64 %92, 1
  store i64 %93, ptr %15, align 4, !tbaa !1
  %94 = load i64, ptr %15, align 4, !tbaa !1
  %95 = add i64 %94, 1
  store i64 %95, ptr %15, align 4, !tbaa !1
  %96 = load i64, ptr %15, align 4, !tbaa !1
  %97 = add i64 %96, 1
  store i64 %97, ptr %15, align 4, !tbaa !1
  %98 = load i64, ptr %15, align 4, !tbaa !1
  %99 = add i64 %98, 1
  store i64 %99, ptr %15, align 4, !tbaa !1
  %100 = load i64, ptr %15, align 4, !tbaa !1
  %101 = add i64 %100, 1
  store i64 %101, ptr %15, align 4, !tbaa !1
  %102 = load i64, ptr %15, align 4, !tbaa !1
  %103 = add i64 %102, 1
  store i64 %103, ptr %15, align 4, !tbaa !1
  %104 = load i64, ptr %15, align 4, !tbaa !1
  %105 = add i64 %104, 1
  store i64 %105, ptr %15, align 4, !tbaa !1
  %106 = load i64, ptr %15, align 4, !tbaa !1
  %107 = add i64 %106, 1
  store i64 %107, ptr %15, align 4, !tbaa !1
  %108 = load i64, ptr %15, align 4, !tbaa !1
  %109 = add i64 %108, 1
  store i64 %109, ptr %15, align 4, !tbaa !1
  %110 = load i64, ptr %15, align 4, !tbaa !1
  %111 = add i64 %110, 1
  store i64 %111, ptr %15, align 4, !tbaa !1
  %112 = load i64, ptr %15, align 4, !tbaa !1
  %113 = add i64 %112, 1
  store i64 %113, ptr %15, align 4, !tbaa !1
  %114 = load i64, ptr %15, align 4, !tbaa !1
  %115 = add i64 %114, 1
  store i64 %115, ptr %15, align 4, !tbaa !1
  %116 = load i64, ptr %15, align 4, !tbaa !1
  %117 = add i64 %116, 1
  store i64 %117, ptr %15, align 4, !tbaa !1
  %118 = load i64, ptr %15, align 4, !tbaa !1
  %119 = add i64 %118, 1
  store i64 %119, ptr %15, align 4, !tbaa !1
  %120 = load i64, ptr %15, align 4, !tbaa !1
  %121 = add i64 %120, 1
  store i64 %121, ptr %15, align 4, !tbaa !1
  %122 = load i64, ptr %15, align 4, !tbaa !1
  %123 = add i64 %122, 1
  store i64 %123, ptr %15, align 4, !tbaa !1
  %124 = load i64, ptr %15, align 4, !tbaa !1
  %125 = add i64 %124, 1
  store i64 %125, ptr %15, align 4, !tbaa !1
  %126 = load i64, ptr %15, align 4, !tbaa !1
  %127 = add i64 %126, 1
  store i64 %127, ptr %15, align 4, !tbaa !1
  %128 = load i64, ptr %15, align 4, !tbaa !1
  %129 = add i64 %128, 1
  store i64 %129, ptr %15, align 4, !tbaa !1
  %130 = load i64, ptr %15, align 4, !tbaa !1
  %131 = add i64 %130, 1
  store i64 %131, ptr %15, align 4, !tbaa !1
  %132 = load i64, ptr %15, align 4, !tbaa !1
  %133 = add i64 %132, 1
  store i64 %133, ptr %15, align 4, !tbaa !1
  %134 = load i64, ptr %15, align 4, !tbaa !1
  %135 = add i64 %134, 1
  store i64 %135, ptr %15, align 4, !tbaa !1
  %136 = load i64, ptr %15, align 4, !tbaa !1
  %137 = add i64 %136, 1
  store i64 %137, ptr %15, align 4, !tbaa !1
  %138 = load i64, ptr %15, align 4, !tbaa !1
  %139 = add i64 %138, 1
  store i64 %139, ptr %15, align 4, !tbaa !1
  %140 = load i64, ptr %15, align 4, !tbaa !1
  %141 = add i64 %140, 1
  store i64 %141, ptr %15, align 4, !tbaa !1
  %142 = load i64, ptr %15, align 4, !tbaa !1
  %143 = add i64 %142, 1
  store i64 %143, ptr %15, align 4, !tbaa !1
  %144 = load i64, ptr %15, align 4, !tbaa !1
  %145 = add i64 %144, 1
  store i64 %145, ptr %15, align 4, !tbaa !1
  %146 = load i64, ptr %15, align 4, !tbaa !1
  %147 = add i64 %146, 1
  store i64 %147, ptr %15, align 4, !tbaa !1
  %148 = load i64, ptr %15, align 4, !tbaa !1
  %149 = add i64 %148, 1
  store i64 %149, ptr %15, align 4, !tbaa !1
  %150 = load i64, ptr %15, align 4, !tbaa !1
  %151 = add i64 %150, 1
  store i64 %151, ptr %15, align 4, !tbaa !1
  %152 = load i64, ptr %15, align 4, !tbaa !1
  %153 = add i64 %152, 1
  store i64 %153, ptr %15, align 4, !tbaa !1
  %154 = load i64, ptr %15, align 4, !tbaa !1
  %155 = add i64 %154, 1
  store i64 %155, ptr %15, align 4, !tbaa !1
  %156 = load i64, ptr %15, align 4, !tbaa !1
  %157 = add i64 %156, 1
  store i64 %157, ptr %15, align 4, !tbaa !1
  %158 = load i64, ptr %15, align 4, !tbaa !1
  %159 = add i64 %158, 1
  store i64 %159, ptr %15, align 4, !tbaa !1
  %160 = load i64, ptr %15, align 4, !tbaa !1
  %161 = add i64 %160, 1
  store i64 %161, ptr %15, align 4, !tbaa !1
  %162 = load i64, ptr %15, align 4, !tbaa !1
  %163 = add i64 %162, 1
  store i64 %163, ptr %15, align 4, !tbaa !1
  %164 = load i64, ptr %15, align 4, !tbaa !1
  %165 = add i64 %164, 1
  store i64 %165, ptr %15, align 4, !tbaa !1
  %166 = load i64, ptr %15, align 4, !tbaa !1
  %167 = add i64 %166, 1
  store i64 %167, ptr %15, align 4, !tbaa !1
  %168 = load i64, ptr %15, align 4, !tbaa !1
  %169 = add i64 %168, 1
  store i64 %169, ptr %15, align 4, !tbaa !1
  %170 = load i64, ptr %15, align 4, !tbaa !1
  %171 = add i64 %170, 1
  store i64 %171, ptr %15, align 4, !tbaa !1
  %172 = load i64, ptr %15, align 4, !tbaa !1
  %173 = add i64 %172, 1
  store i64 %173, ptr %15, align 4, !tbaa !1
  %174 = load i64, ptr %15, align 4, !tbaa !1
  %175 = add i64 %174, 1
  store i64 %175, ptr %15, align 4, !tbaa !1
  %176 = load i64, ptr %15, align 4, !tbaa !1
  %177 = add i64 %176, 1
  store i64 %177, ptr %15, align 4, !tbaa !1
  %178 = load i64, ptr %15, align 4, !tbaa !1
  %179 = add i64 %178, 1
  store i64 %179, ptr %15, align 4, !tbaa !1
  %180 = load i64, ptr %15, align 4, !tbaa !1
  %181 = add i64 %180, 1
  store i64 %181, ptr %15, align 4, !tbaa !1
  %182 = load i64, ptr %15, align 4, !tbaa !1
  %183 = add i64 %182, 1
  store i64 %183, ptr %15, align 4, !tbaa !1
  %184 = load i64, ptr %15, align 4, !tbaa !1
  %185 = add i64 %184, 1
  store i64 %185, ptr %15, align 4, !tbaa !1
  %186 = load i64, ptr %15, align 4, !tbaa !1
  %187 = add i64 %186, 1
  store i64 %187, ptr %15, align 4, !tbaa !1
  %188 = load i64, ptr %15, align 4, !tbaa !1
  %189 = add i64 %188, 1
  store i64 %189, ptr %15, align 4, !tbaa !1
  %190 = load i64, ptr %15, align 4, !tbaa !1
  %191 = add i64 %190, 1
  store i64 %191, ptr %15, align 4, !tbaa !1
  %192 = load i64, ptr %15, align 4, !tbaa !1
  %193 = add i64 %192, 1
  store i64 %193, ptr %15, align 4, !tbaa !1
  %194 = load i64, ptr %15, align 4, !tbaa !1
  %195 = add i64 %194, 1
  store i64 %195, ptr %15, align 4, !tbaa !1
  %196 = load i64, ptr %15, align 4, !tbaa !1
  %197 = add i64 %196, 1
  store i64 %197, ptr %15, align 4, !tbaa !1
  %198 = load i64, ptr %15, align 4, !tbaa !1
  %199 = add i64 %198, 1
  store i64 %199, ptr %15, align 4, !tbaa !1
  %200 = load i64, ptr %15, align 4, !tbaa !1
  %201 = add i64 %200, 1
  store i64 %201, ptr %15, align 4, !tbaa !1
  %202 = load i64, ptr %15, align 4, !tbaa !1
  %203 = add i64 %202, 1
  store i64 %203, ptr %15, align 4, !tbaa !1
  %204 = load i64, ptr %15, align 4, !tbaa !1
  %205 = add i64 %204, 1
  store i64 %205, ptr %15, align 4, !tbaa !1
  %206 = load i64, ptr %15, align 4, !tbaa !1
  %207 = add i64 %206, 1
  store i64 %207, ptr %15, align 4, !tbaa !1
  %208 = load i64, ptr %15, align 4, !tbaa !1
  %209 = add i64 %208, 1
  store i64 %209, ptr %15, align 4, !tbaa !1
  %210 = load i64, ptr %15, align 4, !tbaa !1
  %211 = add i64 %210, 1
  store i64 %211, ptr %15, align 4, !tbaa !1
  %212 = load i64, ptr %15, align 4, !tbaa !1
  %213 = add i64 %212, 1
  store i64 %213, ptr %15, align 4, !tbaa !1
  %214 = load i64, ptr %15, align 4, !tbaa !1
  %215 = add i64 %214, 1
  store i64 %215, ptr %15, align 4, !tbaa !1
  %216 = load i64, ptr %15, align 4, !tbaa !1
  %217 = add i64 %216, 1
  store i64 %217, ptr %15, align 4, !tbaa !1
  %218 = load i64, ptr %15, align 4, !tbaa !1
  %219 = add i64 %218, 1
  store i64 %219, ptr %15, align 4, !tbaa !1
  %220 = load i64, ptr %15, align 4, !tbaa !1
  %221 = add i64 %220, 1
  store i64 %221, ptr %15, align 4, !tbaa !1
  %222 = load i64, ptr %15, align 4, !tbaa !1
  %223 = add i64 %222, 1
  store i64 %223, ptr %15, align 4, !tbaa !1
  %224 = load i64, ptr %15, align 4, !tbaa !1
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
  %42 = load double, ptr %41, align 8, !tbaa !6
  %43 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %31, 1
  %44 = getelementptr inbounds double, ptr %43, i64 %38
  %45 = load double, ptr %44, align 8, !tbaa !6
  %46 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %12, 1
  %47 = getelementptr inbounds double, ptr %46, i64 %38
  %48 = load double, ptr %47, align 8, !tbaa !6
  %49 = fmul double %42, %45
  %50 = fadd double %48, %49
  %51 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %12, 1
  %52 = getelementptr inbounds double, ptr %51, i64 %38
  store double %50, ptr %52, align 8, !tbaa !6
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
  store i64 0, ptr %9, align 4, !tbaa !1
  call void @__catalyst__rt__device_init(ptr @"/home/pichau/QAgents/.venv/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.so", ptr @LightningSimulator, ptr @"{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}", i64 0, i1 false)
  %10 = call ptr @__catalyst__rt__qubit_allocate_array(i64 8)
  %11 = call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %10, i64 7)
  %12 = load ptr, ptr %11, align 8
  %13 = load i64, ptr %9, align 4, !tbaa !1
  %14 = add i64 %13, 1
  store i64 %14, ptr %9, align 4, !tbaa !1
  %15 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %16 = getelementptr inbounds double, ptr %15, i64 %13
  %17 = load double, ptr %16, align 8, !tbaa !6
  call void @__catalyst__qis__RY(double %17, ptr %12, ptr null)
  %18 = load i64, ptr %9, align 4, !tbaa !1
  %19 = add i64 %18, 1
  store i64 %19, ptr %9, align 4, !tbaa !1
  %20 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %21 = getelementptr inbounds double, ptr %20, i64 %18
  %22 = load double, ptr %21, align 8, !tbaa !6
  call void @__catalyst__qis__RZ(double %22, ptr %12, ptr null)
  %23 = load i64, ptr %9, align 4, !tbaa !1
  %24 = add i64 %23, 1
  store i64 %24, ptr %9, align 4, !tbaa !1
  %25 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %26 = getelementptr inbounds double, ptr %25, i64 %23
  %27 = load double, ptr %26, align 8, !tbaa !6
  call void @__catalyst__qis__RY(double %27, ptr %12, ptr null)
  %28 = load i64, ptr %9, align 4, !tbaa !1
  %29 = add i64 %28, 1
  store i64 %29, ptr %9, align 4, !tbaa !1
  %30 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %31 = getelementptr inbounds double, ptr %30, i64 %28
  %32 = load double, ptr %31, align 8, !tbaa !6
  call void @__catalyst__qis__RZ(double %32, ptr %12, ptr null)
  %33 = call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %10, i64 6)
  %34 = load ptr, ptr %33, align 8
  %35 = load i64, ptr %9, align 4, !tbaa !1
  %36 = add i64 %35, 1
  store i64 %36, ptr %9, align 4, !tbaa !1
  %37 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %38 = getelementptr inbounds double, ptr %37, i64 %35
  %39 = load double, ptr %38, align 8, !tbaa !6
  call void @__catalyst__qis__RY(double %39, ptr %34, ptr null)
  %40 = load i64, ptr %9, align 4, !tbaa !1
  %41 = add i64 %40, 1
  store i64 %41, ptr %9, align 4, !tbaa !1
  %42 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %43 = getelementptr inbounds double, ptr %42, i64 %40
  %44 = load double, ptr %43, align 8, !tbaa !6
  call void @__catalyst__qis__RZ(double %44, ptr %34, ptr null)
  %45 = load i64, ptr %9, align 4, !tbaa !1
  %46 = add i64 %45, 1
  store i64 %46, ptr %9, align 4, !tbaa !1
  %47 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %48 = getelementptr inbounds double, ptr %47, i64 %45
  %49 = load double, ptr %48, align 8, !tbaa !6
  call void @__catalyst__qis__RY(double %49, ptr %34, ptr null)
  %50 = load i64, ptr %9, align 4, !tbaa !1
  %51 = add i64 %50, 1
  store i64 %51, ptr %9, align 4, !tbaa !1
  %52 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %53 = getelementptr inbounds double, ptr %52, i64 %50
  %54 = load double, ptr %53, align 8, !tbaa !6
  call void @__catalyst__qis__RZ(double %54, ptr %34, ptr null)
  %55 = call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %10, i64 5)
  %56 = load ptr, ptr %55, align 8
  %57 = load i64, ptr %9, align 4, !tbaa !1
  %58 = add i64 %57, 1
  store i64 %58, ptr %9, align 4, !tbaa !1
  %59 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %60 = getelementptr inbounds double, ptr %59, i64 %57
  %61 = load double, ptr %60, align 8, !tbaa !6
  call void @__catalyst__qis__RY(double %61, ptr %56, ptr null)
  %62 = load i64, ptr %9, align 4, !tbaa !1
  %63 = add i64 %62, 1
  store i64 %63, ptr %9, align 4, !tbaa !1
  %64 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %65 = getelementptr inbounds double, ptr %64, i64 %62
  %66 = load double, ptr %65, align 8, !tbaa !6
  call void @__catalyst__qis__RZ(double %66, ptr %56, ptr null)
  %67 = load i64, ptr %9, align 4, !tbaa !1
  %68 = add i64 %67, 1
  store i64 %68, ptr %9, align 4, !tbaa !1
  %69 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %70 = getelementptr inbounds double, ptr %69, i64 %67
  %71 = load double, ptr %70, align 8, !tbaa !6
  call void @__catalyst__qis__RY(double %71, ptr %56, ptr null)
  %72 = load i64, ptr %9, align 4, !tbaa !1
  %73 = add i64 %72, 1
  store i64 %73, ptr %9, align 4, !tbaa !1
  %74 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %75 = getelementptr inbounds double, ptr %74, i64 %72
  %76 = load double, ptr %75, align 8, !tbaa !6
  call void @__catalyst__qis__RZ(double %76, ptr %56, ptr null)
  %77 = call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %10, i64 4)
  %78 = load ptr, ptr %77, align 8
  %79 = load i64, ptr %9, align 4, !tbaa !1
  %80 = add i64 %79, 1
  store i64 %80, ptr %9, align 4, !tbaa !1
  %81 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %82 = getelementptr inbounds double, ptr %81, i64 %79
  %83 = load double, ptr %82, align 8, !tbaa !6
  call void @__catalyst__qis__RY(double %83, ptr %78, ptr null)
  %84 = load i64, ptr %9, align 4, !tbaa !1
  %85 = add i64 %84, 1
  store i64 %85, ptr %9, align 4, !tbaa !1
  %86 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %87 = getelementptr inbounds double, ptr %86, i64 %84
  %88 = load double, ptr %87, align 8, !tbaa !6
  call void @__catalyst__qis__RZ(double %88, ptr %78, ptr null)
  %89 = load i64, ptr %9, align 4, !tbaa !1
  %90 = add i64 %89, 1
  store i64 %90, ptr %9, align 4, !tbaa !1
  %91 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %92 = getelementptr inbounds double, ptr %91, i64 %89
  %93 = load double, ptr %92, align 8, !tbaa !6
  call void @__catalyst__qis__RY(double %93, ptr %78, ptr null)
  %94 = load i64, ptr %9, align 4, !tbaa !1
  %95 = add i64 %94, 1
  store i64 %95, ptr %9, align 4, !tbaa !1
  %96 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %97 = getelementptr inbounds double, ptr %96, i64 %94
  %98 = load double, ptr %97, align 8, !tbaa !6
  call void @__catalyst__qis__RZ(double %98, ptr %78, ptr null)
  %99 = call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %10, i64 3)
  %100 = load ptr, ptr %99, align 8
  %101 = load i64, ptr %9, align 4, !tbaa !1
  %102 = add i64 %101, 1
  store i64 %102, ptr %9, align 4, !tbaa !1
  %103 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %104 = getelementptr inbounds double, ptr %103, i64 %101
  %105 = load double, ptr %104, align 8, !tbaa !6
  call void @__catalyst__qis__RY(double %105, ptr %100, ptr null)
  %106 = load i64, ptr %9, align 4, !tbaa !1
  %107 = add i64 %106, 1
  store i64 %107, ptr %9, align 4, !tbaa !1
  %108 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %109 = getelementptr inbounds double, ptr %108, i64 %106
  %110 = load double, ptr %109, align 8, !tbaa !6
  call void @__catalyst__qis__RZ(double %110, ptr %100, ptr null)
  %111 = load i64, ptr %9, align 4, !tbaa !1
  %112 = add i64 %111, 1
  store i64 %112, ptr %9, align 4, !tbaa !1
  %113 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %114 = getelementptr inbounds double, ptr %113, i64 %111
  %115 = load double, ptr %114, align 8, !tbaa !6
  call void @__catalyst__qis__RY(double %115, ptr %100, ptr null)
  %116 = load i64, ptr %9, align 4, !tbaa !1
  %117 = add i64 %116, 1
  store i64 %117, ptr %9, align 4, !tbaa !1
  %118 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %119 = getelementptr inbounds double, ptr %118, i64 %116
  %120 = load double, ptr %119, align 8, !tbaa !6
  call void @__catalyst__qis__RZ(double %120, ptr %100, ptr null)
  %121 = call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %10, i64 2)
  %122 = load ptr, ptr %121, align 8
  %123 = load i64, ptr %9, align 4, !tbaa !1
  %124 = add i64 %123, 1
  store i64 %124, ptr %9, align 4, !tbaa !1
  %125 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %126 = getelementptr inbounds double, ptr %125, i64 %123
  %127 = load double, ptr %126, align 8, !tbaa !6
  call void @__catalyst__qis__RY(double %127, ptr %122, ptr null)
  %128 = load i64, ptr %9, align 4, !tbaa !1
  %129 = add i64 %128, 1
  store i64 %129, ptr %9, align 4, !tbaa !1
  %130 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %131 = getelementptr inbounds double, ptr %130, i64 %128
  %132 = load double, ptr %131, align 8, !tbaa !6
  call void @__catalyst__qis__RZ(double %132, ptr %122, ptr null)
  %133 = load i64, ptr %9, align 4, !tbaa !1
  %134 = add i64 %133, 1
  store i64 %134, ptr %9, align 4, !tbaa !1
  %135 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %136 = getelementptr inbounds double, ptr %135, i64 %133
  %137 = load double, ptr %136, align 8, !tbaa !6
  call void @__catalyst__qis__RY(double %137, ptr %122, ptr null)
  %138 = load i64, ptr %9, align 4, !tbaa !1
  %139 = add i64 %138, 1
  store i64 %139, ptr %9, align 4, !tbaa !1
  %140 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %141 = getelementptr inbounds double, ptr %140, i64 %138
  %142 = load double, ptr %141, align 8, !tbaa !6
  call void @__catalyst__qis__RZ(double %142, ptr %122, ptr null)
  %143 = call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %10, i64 0)
  %144 = load ptr, ptr %143, align 8
  %145 = load i64, ptr %9, align 4, !tbaa !1
  %146 = add i64 %145, 1
  store i64 %146, ptr %9, align 4, !tbaa !1
  %147 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %148 = getelementptr inbounds double, ptr %147, i64 %145
  %149 = load double, ptr %148, align 8, !tbaa !6
  call void @__catalyst__qis__RY(double %149, ptr %144, ptr null)
  %150 = load i64, ptr %9, align 4, !tbaa !1
  %151 = add i64 %150, 1
  store i64 %151, ptr %9, align 4, !tbaa !1
  %152 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %153 = getelementptr inbounds double, ptr %152, i64 %150
  %154 = load double, ptr %153, align 8, !tbaa !6
  call void @__catalyst__qis__RZ(double %154, ptr %144, ptr null)
  %155 = load i64, ptr %9, align 4, !tbaa !1
  %156 = add i64 %155, 1
  store i64 %156, ptr %9, align 4, !tbaa !1
  %157 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %158 = getelementptr inbounds double, ptr %157, i64 %155
  %159 = load double, ptr %158, align 8, !tbaa !6
  call void @__catalyst__qis__RY(double %159, ptr %144, ptr null)
  %160 = load i64, ptr %9, align 4, !tbaa !1
  %161 = add i64 %160, 1
  store i64 %161, ptr %9, align 4, !tbaa !1
  %162 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %163 = getelementptr inbounds double, ptr %162, i64 %160
  %164 = load double, ptr %163, align 8, !tbaa !6
  call void @__catalyst__qis__RZ(double %164, ptr %144, ptr null)
  %165 = call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %10, i64 1)
  %166 = load ptr, ptr %165, align 8
  %167 = load i64, ptr %9, align 4, !tbaa !1
  %168 = add i64 %167, 1
  store i64 %168, ptr %9, align 4, !tbaa !1
  %169 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %170 = getelementptr inbounds double, ptr %169, i64 %167
  %171 = load double, ptr %170, align 8, !tbaa !6
  call void @__catalyst__qis__RY(double %171, ptr %166, ptr null)
  %172 = load i64, ptr %9, align 4, !tbaa !1
  %173 = add i64 %172, 1
  store i64 %173, ptr %9, align 4, !tbaa !1
  %174 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %175 = getelementptr inbounds double, ptr %174, i64 %172
  %176 = load double, ptr %175, align 8, !tbaa !6
  call void @__catalyst__qis__RZ(double %176, ptr %166, ptr null)
  %177 = load i64, ptr %9, align 4, !tbaa !1
  %178 = add i64 %177, 1
  store i64 %178, ptr %9, align 4, !tbaa !1
  %179 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %180 = getelementptr inbounds double, ptr %179, i64 %177
  %181 = load double, ptr %180, align 8, !tbaa !6
  call void @__catalyst__qis__RY(double %181, ptr %166, ptr null)
  %182 = load i64, ptr %9, align 4, !tbaa !1
  %183 = add i64 %182, 1
  store i64 %183, ptr %9, align 4, !tbaa !1
  %184 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %185 = getelementptr inbounds double, ptr %184, i64 %182
  %186 = load double, ptr %185, align 8, !tbaa !6
  call void @__catalyst__qis__RZ(double %186, ptr %166, ptr null)
  call void @__catalyst__qis__CNOT(ptr %144, ptr %166, ptr null)
  call void @__catalyst__qis__CNOT(ptr %166, ptr %122, ptr null)
  call void @__catalyst__qis__CNOT(ptr %122, ptr %100, ptr null)
  call void @__catalyst__qis__CNOT(ptr %100, ptr %78, ptr null)
  call void @__catalyst__qis__CNOT(ptr %78, ptr %56, ptr null)
  call void @__catalyst__qis__CNOT(ptr %56, ptr %34, ptr null)
  call void @__catalyst__qis__CNOT(ptr %34, ptr %12, ptr null)
  %187 = load i64, ptr %9, align 4, !tbaa !1
  %188 = add i64 %187, 1
  store i64 %188, ptr %9, align 4, !tbaa !1
  %189 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %190 = getelementptr inbounds double, ptr %189, i64 %187
  %191 = load double, ptr %190, align 8, !tbaa !6
  call void @__catalyst__qis__RZ(double %191, ptr %34, ptr null)
  %192 = load i64, ptr %9, align 4, !tbaa !1
  %193 = add i64 %192, 1
  store i64 %193, ptr %9, align 4, !tbaa !1
  %194 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %195 = getelementptr inbounds double, ptr %194, i64 %192
  %196 = load double, ptr %195, align 8, !tbaa !6
  call void @__catalyst__qis__RY(double %196, ptr %34, ptr null)
  %197 = load i64, ptr %9, align 4, !tbaa !1
  %198 = add i64 %197, 1
  store i64 %198, ptr %9, align 4, !tbaa !1
  %199 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %200 = getelementptr inbounds double, ptr %199, i64 %197
  %201 = load double, ptr %200, align 8, !tbaa !6
  call void @__catalyst__qis__RZ(double %201, ptr %34, ptr null)
  %202 = load i64, ptr %9, align 4, !tbaa !1
  %203 = add i64 %202, 1
  store i64 %203, ptr %9, align 4, !tbaa !1
  %204 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %205 = getelementptr inbounds double, ptr %204, i64 %202
  %206 = load double, ptr %205, align 8, !tbaa !6
  call void @__catalyst__qis__RZ(double %206, ptr %78, ptr null)
  %207 = load i64, ptr %9, align 4, !tbaa !1
  %208 = add i64 %207, 1
  store i64 %208, ptr %9, align 4, !tbaa !1
  %209 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %210 = getelementptr inbounds double, ptr %209, i64 %207
  %211 = load double, ptr %210, align 8, !tbaa !6
  call void @__catalyst__qis__RY(double %211, ptr %78, ptr null)
  %212 = load i64, ptr %9, align 4, !tbaa !1
  %213 = add i64 %212, 1
  store i64 %213, ptr %9, align 4, !tbaa !1
  %214 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %215 = getelementptr inbounds double, ptr %214, i64 %212
  %216 = load double, ptr %215, align 8, !tbaa !6
  call void @__catalyst__qis__RZ(double %216, ptr %78, ptr null)
  call void @__catalyst__qis__CNOT(ptr %12, ptr %144, ptr null)
  %217 = load i64, ptr %9, align 4, !tbaa !1
  %218 = add i64 %217, 1
  store i64 %218, ptr %9, align 4, !tbaa !1
  %219 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %220 = getelementptr inbounds double, ptr %219, i64 %217
  %221 = load double, ptr %220, align 8, !tbaa !6
  call void @__catalyst__qis__RZ(double %221, ptr %144, ptr null)
  %222 = load i64, ptr %9, align 4, !tbaa !1
  %223 = add i64 %222, 1
  store i64 %223, ptr %9, align 4, !tbaa !1
  %224 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %225 = getelementptr inbounds double, ptr %224, i64 %222
  %226 = load double, ptr %225, align 8, !tbaa !6
  call void @__catalyst__qis__RY(double %226, ptr %144, ptr null)
  %227 = load i64, ptr %9, align 4, !tbaa !1
  %228 = add i64 %227, 1
  store i64 %228, ptr %9, align 4, !tbaa !1
  %229 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %230 = getelementptr inbounds double, ptr %229, i64 %227
  %231 = load double, ptr %230, align 8, !tbaa !6
  call void @__catalyst__qis__RZ(double %231, ptr %144, ptr null)
  %232 = load i64, ptr %9, align 4, !tbaa !1
  %233 = add i64 %232, 1
  store i64 %233, ptr %9, align 4, !tbaa !1
  %234 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %235 = getelementptr inbounds double, ptr %234, i64 %232
  %236 = load double, ptr %235, align 8, !tbaa !6
  call void @__catalyst__qis__RZ(double %236, ptr %122, ptr null)
  %237 = load i64, ptr %9, align 4, !tbaa !1
  %238 = add i64 %237, 1
  store i64 %238, ptr %9, align 4, !tbaa !1
  %239 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %240 = getelementptr inbounds double, ptr %239, i64 %237
  %241 = load double, ptr %240, align 8, !tbaa !6
  call void @__catalyst__qis__RY(double %241, ptr %122, ptr null)
  %242 = load i64, ptr %9, align 4, !tbaa !1
  %243 = add i64 %242, 1
  store i64 %243, ptr %9, align 4, !tbaa !1
  %244 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %245 = getelementptr inbounds double, ptr %244, i64 %242
  %246 = load double, ptr %245, align 8, !tbaa !6
  call void @__catalyst__qis__RZ(double %246, ptr %122, ptr null)
  call void @__catalyst__qis__CNOT(ptr %144, ptr %122, ptr null)
  call void @__catalyst__qis__CNOT(ptr %122, ptr %78, ptr null)
  call void @__catalyst__qis__CNOT(ptr %78, ptr %34, ptr null)
  call void @__catalyst__qis__CNOT(ptr %34, ptr %144, ptr null)
  %247 = load i64, ptr %9, align 4, !tbaa !1
  %248 = add i64 %247, 1
  store i64 %248, ptr %9, align 4, !tbaa !1
  %249 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %250 = getelementptr inbounds double, ptr %249, i64 %247
  %251 = load double, ptr %250, align 8, !tbaa !6
  call void @__catalyst__qis__RZ(double %251, ptr %144, ptr null)
  %252 = load i64, ptr %9, align 4, !tbaa !1
  %253 = add i64 %252, 1
  store i64 %253, ptr %9, align 4, !tbaa !1
  %254 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %255 = getelementptr inbounds double, ptr %254, i64 %252
  %256 = load double, ptr %255, align 8, !tbaa !6
  call void @__catalyst__qis__RY(double %256, ptr %144, ptr null)
  %257 = load i64, ptr %9, align 4, !tbaa !1
  %258 = add i64 %257, 1
  store i64 %258, ptr %9, align 4, !tbaa !1
  %259 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %260 = getelementptr inbounds double, ptr %259, i64 %257
  %261 = load double, ptr %260, align 8, !tbaa !6
  call void @__catalyst__qis__RZ(double %261, ptr %144, ptr null)
  %262 = load i64, ptr %9, align 4, !tbaa !1
  %263 = add i64 %262, 1
  store i64 %263, ptr %9, align 4, !tbaa !1
  %264 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %265 = getelementptr inbounds double, ptr %264, i64 %262
  %266 = load double, ptr %265, align 8, !tbaa !6
  call void @__catalyst__qis__RZ(double %266, ptr %56, ptr null)
  %267 = load i64, ptr %9, align 4, !tbaa !1
  %268 = add i64 %267, 1
  store i64 %268, ptr %9, align 4, !tbaa !1
  %269 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %270 = getelementptr inbounds double, ptr %269, i64 %267
  %271 = load double, ptr %270, align 8, !tbaa !6
  call void @__catalyst__qis__RY(double %271, ptr %56, ptr null)
  %272 = load i64, ptr %9, align 4, !tbaa !1
  %273 = add i64 %272, 1
  store i64 %273, ptr %9, align 4, !tbaa !1
  %274 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %275 = getelementptr inbounds double, ptr %274, i64 %272
  %276 = load double, ptr %275, align 8, !tbaa !6
  call void @__catalyst__qis__RZ(double %276, ptr %56, ptr null)
  %277 = load i64, ptr %9, align 4, !tbaa !1
  %278 = add i64 %277, 1
  store i64 %278, ptr %9, align 4, !tbaa !1
  %279 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %280 = getelementptr inbounds double, ptr %279, i64 %277
  %281 = load double, ptr %280, align 8, !tbaa !6
  call void @__catalyst__qis__RZ(double %281, ptr %166, ptr null)
  %282 = load i64, ptr %9, align 4, !tbaa !1
  %283 = add i64 %282, 1
  store i64 %283, ptr %9, align 4, !tbaa !1
  %284 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %285 = getelementptr inbounds double, ptr %284, i64 %282
  %286 = load double, ptr %285, align 8, !tbaa !6
  call void @__catalyst__qis__RY(double %286, ptr %166, ptr null)
  %287 = load i64, ptr %9, align 4, !tbaa !1
  %288 = add i64 %287, 1
  store i64 %288, ptr %9, align 4, !tbaa !1
  %289 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %290 = getelementptr inbounds double, ptr %289, i64 %287
  %291 = load double, ptr %290, align 8, !tbaa !6
  call void @__catalyst__qis__RZ(double %291, ptr %166, ptr null)
  %292 = load i64, ptr %9, align 4, !tbaa !1
  %293 = add i64 %292, 1
  store i64 %293, ptr %9, align 4, !tbaa !1
  %294 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %295 = getelementptr inbounds double, ptr %294, i64 %292
  %296 = load double, ptr %295, align 8, !tbaa !6
  call void @__catalyst__qis__RZ(double %296, ptr %100, ptr null)
  %297 = load i64, ptr %9, align 4, !tbaa !1
  %298 = add i64 %297, 1
  store i64 %298, ptr %9, align 4, !tbaa !1
  %299 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %300 = getelementptr inbounds double, ptr %299, i64 %297
  %301 = load double, ptr %300, align 8, !tbaa !6
  call void @__catalyst__qis__RY(double %301, ptr %100, ptr null)
  %302 = load i64, ptr %9, align 4, !tbaa !1
  %303 = add i64 %302, 1
  store i64 %303, ptr %9, align 4, !tbaa !1
  %304 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %305 = getelementptr inbounds double, ptr %304, i64 %302
  %306 = load double, ptr %305, align 8, !tbaa !6
  call void @__catalyst__qis__RZ(double %306, ptr %100, ptr null)
  call void @__catalyst__qis__CNOT(ptr %166, ptr %100, ptr null)
  call void @__catalyst__qis__CNOT(ptr %100, ptr %56, ptr null)
  %307 = load i64, ptr %9, align 4, !tbaa !1
  %308 = add i64 %307, 1
  store i64 %308, ptr %9, align 4, !tbaa !1
  %309 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %310 = getelementptr inbounds double, ptr %309, i64 %307
  %311 = load double, ptr %310, align 8, !tbaa !6
  call void @__catalyst__qis__RZ(double %311, ptr %100, ptr null)
  %312 = load i64, ptr %9, align 4, !tbaa !1
  %313 = add i64 %312, 1
  store i64 %313, ptr %9, align 4, !tbaa !1
  %314 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %315 = getelementptr inbounds double, ptr %314, i64 %312
  %316 = load double, ptr %315, align 8, !tbaa !6
  call void @__catalyst__qis__RY(double %316, ptr %100, ptr null)
  %317 = load i64, ptr %9, align 4, !tbaa !1
  %318 = add i64 %317, 1
  store i64 %318, ptr %9, align 4, !tbaa !1
  %319 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %320 = getelementptr inbounds double, ptr %319, i64 %317
  %321 = load double, ptr %320, align 8, !tbaa !6
  call void @__catalyst__qis__RZ(double %321, ptr %100, ptr null)
  call void @__catalyst__qis__CNOT(ptr %144, ptr %100, ptr null)
  %322 = load i64, ptr %9, align 4, !tbaa !1
  %323 = add i64 %322, 1
  store i64 %323, ptr %9, align 4, !tbaa !1
  %324 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %325 = getelementptr inbounds double, ptr %324, i64 %322
  %326 = load double, ptr %325, align 8, !tbaa !6
  call void @__catalyst__qis__RZ(double %326, ptr %122, ptr null)
  %327 = load i64, ptr %9, align 4, !tbaa !1
  %328 = add i64 %327, 1
  store i64 %328, ptr %9, align 4, !tbaa !1
  %329 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %330 = getelementptr inbounds double, ptr %329, i64 %327
  %331 = load double, ptr %330, align 8, !tbaa !6
  call void @__catalyst__qis__RY(double %331, ptr %122, ptr null)
  %332 = load i64, ptr %9, align 4, !tbaa !1
  %333 = add i64 %332, 1
  store i64 %333, ptr %9, align 4, !tbaa !1
  %334 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %335 = getelementptr inbounds double, ptr %334, i64 %332
  %336 = load double, ptr %335, align 8, !tbaa !6
  call void @__catalyst__qis__RZ(double %336, ptr %122, ptr null)
  %337 = load i64, ptr %9, align 4, !tbaa !1
  %338 = add i64 %337, 1
  store i64 %338, ptr %9, align 4, !tbaa !1
  %339 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %340 = getelementptr inbounds double, ptr %339, i64 %337
  %341 = load double, ptr %340, align 8, !tbaa !6
  call void @__catalyst__qis__RZ(double %341, ptr %12, ptr null)
  %342 = load i64, ptr %9, align 4, !tbaa !1
  %343 = add i64 %342, 1
  store i64 %343, ptr %9, align 4, !tbaa !1
  %344 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %345 = getelementptr inbounds double, ptr %344, i64 %342
  %346 = load double, ptr %345, align 8, !tbaa !6
  call void @__catalyst__qis__RY(double %346, ptr %12, ptr null)
  %347 = load i64, ptr %9, align 4, !tbaa !1
  %348 = add i64 %347, 1
  store i64 %348, ptr %9, align 4, !tbaa !1
  %349 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %350 = getelementptr inbounds double, ptr %349, i64 %347
  %351 = load double, ptr %350, align 8, !tbaa !6
  call void @__catalyst__qis__RZ(double %351, ptr %12, ptr null)
  call void @__catalyst__qis__CNOT(ptr %56, ptr %12, ptr null)
  %352 = load i64, ptr %9, align 4, !tbaa !1
  %353 = add i64 %352, 1
  store i64 %353, ptr %9, align 4, !tbaa !1
  %354 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %355 = getelementptr inbounds double, ptr %354, i64 %352
  %356 = load double, ptr %355, align 8, !tbaa !6
  call void @__catalyst__qis__RZ(double %356, ptr %56, ptr null)
  %357 = load i64, ptr %9, align 4, !tbaa !1
  %358 = add i64 %357, 1
  store i64 %358, ptr %9, align 4, !tbaa !1
  %359 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %360 = getelementptr inbounds double, ptr %359, i64 %357
  %361 = load double, ptr %360, align 8, !tbaa !6
  call void @__catalyst__qis__RY(double %361, ptr %56, ptr null)
  %362 = load i64, ptr %9, align 4, !tbaa !1
  %363 = add i64 %362, 1
  store i64 %363, ptr %9, align 4, !tbaa !1
  %364 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %365 = getelementptr inbounds double, ptr %364, i64 %362
  %366 = load double, ptr %365, align 8, !tbaa !6
  call void @__catalyst__qis__RZ(double %366, ptr %56, ptr null)
  call void @__catalyst__qis__CNOT(ptr %122, ptr %56, ptr null)
  call void @__catalyst__qis__CNOT(ptr %56, ptr %144, ptr null)
  %367 = load i64, ptr %9, align 4, !tbaa !1
  %368 = add i64 %367, 1
  store i64 %368, ptr %9, align 4, !tbaa !1
  %369 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %370 = getelementptr inbounds double, ptr %369, i64 %367
  %371 = load double, ptr %370, align 8, !tbaa !6
  call void @__catalyst__qis__RZ(double %371, ptr %144, ptr null)
  %372 = load i64, ptr %9, align 4, !tbaa !1
  %373 = add i64 %372, 1
  store i64 %373, ptr %9, align 4, !tbaa !1
  %374 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %375 = getelementptr inbounds double, ptr %374, i64 %372
  %376 = load double, ptr %375, align 8, !tbaa !6
  call void @__catalyst__qis__RY(double %376, ptr %144, ptr null)
  %377 = load i64, ptr %9, align 4, !tbaa !1
  %378 = add i64 %377, 1
  store i64 %378, ptr %9, align 4, !tbaa !1
  %379 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %380 = getelementptr inbounds double, ptr %379, i64 %377
  %381 = load double, ptr %380, align 8, !tbaa !6
  call void @__catalyst__qis__RZ(double %381, ptr %144, ptr null)
  call void @__catalyst__qis__CNOT(ptr %12, ptr %166, ptr null)
  %382 = load i64, ptr %9, align 4, !tbaa !1
  %383 = add i64 %382, 1
  store i64 %383, ptr %9, align 4, !tbaa !1
  %384 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %385 = getelementptr inbounds double, ptr %384, i64 %382
  %386 = load double, ptr %385, align 8, !tbaa !6
  call void @__catalyst__qis__RZ(double %386, ptr %12, ptr null)
  %387 = load i64, ptr %9, align 4, !tbaa !1
  %388 = add i64 %387, 1
  store i64 %388, ptr %9, align 4, !tbaa !1
  %389 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %390 = getelementptr inbounds double, ptr %389, i64 %387
  %391 = load double, ptr %390, align 8, !tbaa !6
  call void @__catalyst__qis__RY(double %391, ptr %12, ptr null)
  %392 = load i64, ptr %9, align 4, !tbaa !1
  %393 = add i64 %392, 1
  store i64 %393, ptr %9, align 4, !tbaa !1
  %394 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %395 = getelementptr inbounds double, ptr %394, i64 %392
  %396 = load double, ptr %395, align 8, !tbaa !6
  call void @__catalyst__qis__RZ(double %396, ptr %12, ptr null)
  %397 = load i64, ptr %9, align 4, !tbaa !1
  %398 = add i64 %397, 1
  store i64 %398, ptr %9, align 4, !tbaa !1
  %399 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %400 = getelementptr inbounds double, ptr %399, i64 %397
  %401 = load double, ptr %400, align 8, !tbaa !6
  call void @__catalyst__qis__RZ(double %401, ptr %166, ptr null)
  %402 = load i64, ptr %9, align 4, !tbaa !1
  %403 = add i64 %402, 1
  store i64 %403, ptr %9, align 4, !tbaa !1
  %404 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %405 = getelementptr inbounds double, ptr %404, i64 %402
  %406 = load double, ptr %405, align 8, !tbaa !6
  call void @__catalyst__qis__RY(double %406, ptr %166, ptr null)
  %407 = load i64, ptr %9, align 4, !tbaa !1
  %408 = add i64 %407, 1
  store i64 %408, ptr %9, align 4, !tbaa !1
  %409 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %410 = getelementptr inbounds double, ptr %409, i64 %407
  %411 = load double, ptr %410, align 8, !tbaa !6
  call void @__catalyst__qis__RZ(double %411, ptr %166, ptr null)
  %412 = load i64, ptr %9, align 4, !tbaa !1
  %413 = add i64 %412, 1
  store i64 %413, ptr %9, align 4, !tbaa !1
  %414 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %415 = getelementptr inbounds double, ptr %414, i64 %412
  %416 = load double, ptr %415, align 8, !tbaa !6
  call void @__catalyst__qis__RZ(double %416, ptr %78, ptr null)
  %417 = load i64, ptr %9, align 4, !tbaa !1
  %418 = add i64 %417, 1
  store i64 %418, ptr %9, align 4, !tbaa !1
  %419 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %420 = getelementptr inbounds double, ptr %419, i64 %417
  %421 = load double, ptr %420, align 8, !tbaa !6
  call void @__catalyst__qis__RY(double %421, ptr %78, ptr null)
  %422 = load i64, ptr %9, align 4, !tbaa !1
  %423 = add i64 %422, 1
  store i64 %423, ptr %9, align 4, !tbaa !1
  %424 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %425 = getelementptr inbounds double, ptr %424, i64 %422
  %426 = load double, ptr %425, align 8, !tbaa !6
  call void @__catalyst__qis__RZ(double %426, ptr %78, ptr null)
  call void @__catalyst__qis__CNOT(ptr %166, ptr %78, ptr null)
  call void @__catalyst__qis__CNOT(ptr %78, ptr %12, ptr null)
  %427 = load i64, ptr %9, align 4, !tbaa !1
  %428 = add i64 %427, 1
  store i64 %428, ptr %9, align 4, !tbaa !1
  %429 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %430 = getelementptr inbounds double, ptr %429, i64 %427
  %431 = load double, ptr %430, align 8, !tbaa !6
  call void @__catalyst__qis__RZ(double %431, ptr %78, ptr null)
  %432 = load i64, ptr %9, align 4, !tbaa !1
  %433 = add i64 %432, 1
  store i64 %433, ptr %9, align 4, !tbaa !1
  %434 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %435 = getelementptr inbounds double, ptr %434, i64 %432
  %436 = load double, ptr %435, align 8, !tbaa !6
  call void @__catalyst__qis__RY(double %436, ptr %78, ptr null)
  %437 = load i64, ptr %9, align 4, !tbaa !1
  %438 = add i64 %437, 1
  store i64 %438, ptr %9, align 4, !tbaa !1
  %439 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %440 = getelementptr inbounds double, ptr %439, i64 %437
  %441 = load double, ptr %440, align 8, !tbaa !6
  call void @__catalyst__qis__RZ(double %441, ptr %78, ptr null)
  call void @__catalyst__qis__CNOT(ptr %144, ptr %78, ptr null)
  call void @__catalyst__qis__CNOT(ptr %78, ptr %144, ptr null)
  %442 = load i64, ptr %9, align 4, !tbaa !1
  %443 = add i64 %442, 1
  store i64 %443, ptr %9, align 4, !tbaa !1
  %444 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %445 = getelementptr inbounds double, ptr %444, i64 %442
  %446 = load double, ptr %445, align 8, !tbaa !6
  call void @__catalyst__qis__RZ(double %446, ptr %34, ptr null)
  %447 = load i64, ptr %9, align 4, !tbaa !1
  %448 = add i64 %447, 1
  store i64 %448, ptr %9, align 4, !tbaa !1
  %449 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %450 = getelementptr inbounds double, ptr %449, i64 %447
  %451 = load double, ptr %450, align 8, !tbaa !6
  call void @__catalyst__qis__RY(double %451, ptr %34, ptr null)
  %452 = load i64, ptr %9, align 4, !tbaa !1
  %453 = add i64 %452, 1
  store i64 %453, ptr %9, align 4, !tbaa !1
  %454 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %455 = getelementptr inbounds double, ptr %454, i64 %452
  %456 = load double, ptr %455, align 8, !tbaa !6
  call void @__catalyst__qis__RZ(double %456, ptr %34, ptr null)
  call void @__catalyst__qis__CNOT(ptr %100, ptr %34, ptr null)
  call void @__catalyst__qis__CNOT(ptr %34, ptr %166, ptr null)
  %457 = load i64, ptr %9, align 4, !tbaa !1
  %458 = add i64 %457, 1
  store i64 %458, ptr %9, align 4, !tbaa !1
  %459 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %460 = getelementptr inbounds double, ptr %459, i64 %457
  %461 = load double, ptr %460, align 8, !tbaa !6
  call void @__catalyst__qis__RZ(double %461, ptr %166, ptr null)
  %462 = load i64, ptr %9, align 4, !tbaa !1
  %463 = add i64 %462, 1
  store i64 %463, ptr %9, align 4, !tbaa !1
  %464 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %465 = getelementptr inbounds double, ptr %464, i64 %462
  %466 = load double, ptr %465, align 8, !tbaa !6
  call void @__catalyst__qis__RY(double %466, ptr %166, ptr null)
  %467 = load i64, ptr %9, align 4, !tbaa !1
  %468 = add i64 %467, 1
  store i64 %468, ptr %9, align 4, !tbaa !1
  %469 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %470 = getelementptr inbounds double, ptr %469, i64 %467
  %471 = load double, ptr %470, align 8, !tbaa !6
  call void @__catalyst__qis__RZ(double %471, ptr %166, ptr null)
  %472 = load i64, ptr %9, align 4, !tbaa !1
  %473 = add i64 %472, 1
  store i64 %473, ptr %9, align 4, !tbaa !1
  %474 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %475 = getelementptr inbounds double, ptr %474, i64 %472
  %476 = load double, ptr %475, align 8, !tbaa !6
  call void @__catalyst__qis__RZ(double %476, ptr %56, ptr null)
  %477 = load i64, ptr %9, align 4, !tbaa !1
  %478 = add i64 %477, 1
  store i64 %478, ptr %9, align 4, !tbaa !1
  %479 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %480 = getelementptr inbounds double, ptr %479, i64 %477
  %481 = load double, ptr %480, align 8, !tbaa !6
  call void @__catalyst__qis__RY(double %481, ptr %56, ptr null)
  %482 = load i64, ptr %9, align 4, !tbaa !1
  %483 = add i64 %482, 1
  store i64 %483, ptr %9, align 4, !tbaa !1
  %484 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %485 = getelementptr inbounds double, ptr %484, i64 %482
  %486 = load double, ptr %485, align 8, !tbaa !6
  call void @__catalyst__qis__RZ(double %486, ptr %56, ptr null)
  call void @__catalyst__qis__CNOT(ptr %166, ptr %56, ptr null)
  call void @__catalyst__qis__CNOT(ptr %56, ptr %166, ptr null)
  call void @__catalyst__qis__CNOT(ptr %12, ptr %122, ptr null)
  %487 = load i64, ptr %9, align 4, !tbaa !1
  %488 = add i64 %487, 1
  store i64 %488, ptr %9, align 4, !tbaa !1
  %489 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %490 = getelementptr inbounds double, ptr %489, i64 %487
  %491 = load double, ptr %490, align 8, !tbaa !6
  call void @__catalyst__qis__RZ(double %491, ptr %122, ptr null)
  %492 = load i64, ptr %9, align 4, !tbaa !1
  %493 = add i64 %492, 1
  store i64 %493, ptr %9, align 4, !tbaa !1
  %494 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %495 = getelementptr inbounds double, ptr %494, i64 %492
  %496 = load double, ptr %495, align 8, !tbaa !6
  call void @__catalyst__qis__RY(double %496, ptr %122, ptr null)
  %497 = load i64, ptr %9, align 4, !tbaa !1
  %498 = add i64 %497, 1
  store i64 %498, ptr %9, align 4, !tbaa !1
  %499 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %500 = getelementptr inbounds double, ptr %499, i64 %497
  %501 = load double, ptr %500, align 8, !tbaa !6
  call void @__catalyst__qis__RZ(double %501, ptr %122, ptr null)
  %502 = load i64, ptr %9, align 4, !tbaa !1
  %503 = add i64 %502, 1
  store i64 %503, ptr %9, align 4, !tbaa !1
  %504 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %505 = getelementptr inbounds double, ptr %504, i64 %502
  %506 = load double, ptr %505, align 8, !tbaa !6
  call void @__catalyst__qis__RZ(double %506, ptr %34, ptr null)
  %507 = load i64, ptr %9, align 4, !tbaa !1
  %508 = add i64 %507, 1
  store i64 %508, ptr %9, align 4, !tbaa !1
  %509 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %510 = getelementptr inbounds double, ptr %509, i64 %507
  %511 = load double, ptr %510, align 8, !tbaa !6
  call void @__catalyst__qis__RY(double %511, ptr %34, ptr null)
  %512 = load i64, ptr %9, align 4, !tbaa !1
  %513 = add i64 %512, 1
  store i64 %513, ptr %9, align 4, !tbaa !1
  %514 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %515 = getelementptr inbounds double, ptr %514, i64 %512
  %516 = load double, ptr %515, align 8, !tbaa !6
  call void @__catalyst__qis__RZ(double %516, ptr %34, ptr null)
  call void @__catalyst__qis__CNOT(ptr %122, ptr %34, ptr null)
  call void @__catalyst__qis__CNOT(ptr %34, ptr %122, ptr null)
  %517 = load i64, ptr %9, align 4, !tbaa !1
  %518 = add i64 %517, 1
  store i64 %518, ptr %9, align 4, !tbaa !1
  %519 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %520 = getelementptr inbounds double, ptr %519, i64 %517
  %521 = load double, ptr %520, align 8, !tbaa !6
  call void @__catalyst__qis__RZ(double %521, ptr %100, ptr null)
  %522 = load i64, ptr %9, align 4, !tbaa !1
  %523 = add i64 %522, 1
  store i64 %523, ptr %9, align 4, !tbaa !1
  %524 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %525 = getelementptr inbounds double, ptr %524, i64 %522
  %526 = load double, ptr %525, align 8, !tbaa !6
  call void @__catalyst__qis__RY(double %526, ptr %100, ptr null)
  %527 = load i64, ptr %9, align 4, !tbaa !1
  %528 = add i64 %527, 1
  store i64 %528, ptr %9, align 4, !tbaa !1
  %529 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %530 = getelementptr inbounds double, ptr %529, i64 %527
  %531 = load double, ptr %530, align 8, !tbaa !6
  call void @__catalyst__qis__RZ(double %531, ptr %100, ptr null)
  %532 = load i64, ptr %9, align 4, !tbaa !1
  %533 = add i64 %532, 1
  store i64 %533, ptr %9, align 4, !tbaa !1
  %534 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %535 = getelementptr inbounds double, ptr %534, i64 %532
  %536 = load double, ptr %535, align 8, !tbaa !6
  call void @__catalyst__qis__RZ(double %536, ptr %12, ptr null)
  %537 = load i64, ptr %9, align 4, !tbaa !1
  %538 = add i64 %537, 1
  store i64 %538, ptr %9, align 4, !tbaa !1
  %539 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %540 = getelementptr inbounds double, ptr %539, i64 %537
  %541 = load double, ptr %540, align 8, !tbaa !6
  call void @__catalyst__qis__RY(double %541, ptr %12, ptr null)
  %542 = load i64, ptr %9, align 4, !tbaa !1
  %543 = add i64 %542, 1
  store i64 %543, ptr %9, align 4, !tbaa !1
  %544 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, 1
  %545 = getelementptr inbounds double, ptr %544, i64 %542
  %546 = load double, ptr %545, align 8, !tbaa !6
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
  store double %548, ptr %554, align 8, !tbaa !6
  call void @__catalyst__rt__qubit_release_array(ptr %10)
  call void @__catalyst__rt__device_release()
  %555 = load double, ptr %554, align 8, !tbaa !6
  %556 = extractvalue { ptr, ptr, i64 } %8, 1
  store double %555, ptr %556, align 8, !tbaa !6
  ret void
}

define ptr @qnode_forward_0.quantum.augfwd(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7) {
  call void @qnode_forward_0.quantum(ptr %0, ptr %2, ptr %4, ptr %6)
  ret ptr null
}

define { ptr, ptr, i64 } @qnode_forward_0.preprocess(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, ptr %9, ptr %10, i64 %11, i64 %12, i64 %13, i64 %14) {
  %16 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %9, 0
  %17 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %16, ptr %10, 1
  %18 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %17, i64 %11, 2
  %19 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %18, i64 %12, 3, 0
  %20 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %19, i64 %13, 4, 0
  %21 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } poison, ptr %0, 0
  %22 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %21, ptr %1, 1
  %23 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %22, i64 %2, 2
  %24 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %23, i64 %3, 3, 0
  %25 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %24, i64 %6, 4, 0
  %26 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %25, i64 %4, 3, 1
  %27 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %26, i64 %7, 4, 1
  %28 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %27, i64 %5, 3, 2
  %29 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %28, i64 %8, 4, 2
  %30 = alloca { ptr, ptr, i64 }, i64 1, align 8
  %31 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, i64 1, align 8
  %32 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, i64 1, align 8
  %33 = alloca { ptr, ptr, i64, [3 x i64], [3 x i64] }, i64 1, align 8
  %34 = getelementptr double, ptr null, i64 %14
  %35 = ptrtoint ptr %34 to i64
  %36 = call ptr @_mlir_memref_to_llvm_alloc(i64 %35)
  %37 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %36, 0
  %38 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %37, ptr %36, 1
  %39 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %38, i64 0, 2
  %40 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %39, i64 %14, 3, 0
  %41 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %40, i64 1, 4, 0
  %42 = alloca i64, i64 1, align 8
  store i64 0, ptr %42, align 4, !tbaa !1
  %43 = getelementptr inbounds float, ptr %1, i32 74
  %44 = load float, ptr %43, align 4, !tbaa !4
  %45 = getelementptr inbounds float, ptr %1, i32 73
  %46 = load float, ptr %45, align 4, !tbaa !4
  %47 = getelementptr inbounds float, ptr %1, i32 72
  %48 = load float, ptr %47, align 4, !tbaa !4
  %49 = getelementptr inbounds float, ptr %1, i32 50
  %50 = load float, ptr %49, align 4, !tbaa !4
  %51 = getelementptr inbounds float, ptr %1, i32 49
  %52 = load float, ptr %51, align 4, !tbaa !4
  %53 = getelementptr inbounds float, ptr %1, i32 48
  %54 = load float, ptr %53, align 4, !tbaa !4
  %55 = getelementptr inbounds float, ptr %1, i32 44
  %56 = load float, ptr %55, align 4, !tbaa !4
  %57 = getelementptr inbounds float, ptr %1, i32 43
  %58 = load float, ptr %57, align 4, !tbaa !4
  %59 = getelementptr inbounds float, ptr %1, i32 42
  %60 = load float, ptr %59, align 4, !tbaa !4
  %61 = getelementptr inbounds float, ptr %1, i32 23
  %62 = load float, ptr %61, align 4, !tbaa !4
  %63 = getelementptr inbounds float, ptr %1, i32 22
  %64 = load float, ptr %63, align 4, !tbaa !4
  %65 = getelementptr inbounds float, ptr %1, i32 21
  %66 = load float, ptr %65, align 4, !tbaa !4
  %67 = call ptr @_mlir_memref_to_llvm_alloc(i64 96)
  %68 = ptrtoint ptr %67 to i64
  %69 = add i64 %68, 63
  %70 = urem i64 %69, 64
  %71 = sub i64 %69, %70
  %72 = inttoptr i64 %71 to ptr
  br label %73

73:                                               ; preds = %76, %15
  %74 = phi i64 [ %79, %76 ], [ 0, %15 ]
  %75 = icmp slt i64 %74, 8
  br i1 %75, label %76, label %80

76:                                               ; preds = %73
  %77 = load float, ptr @__constant_xf32, align 4, !tbaa !4
  %78 = getelementptr inbounds float, ptr %72, i64 %74
  store float %77, ptr %78, align 4, !tbaa !4
  %79 = add i64 %74, 1
  br label %73

80:                                               ; preds = %73
  br label %81

81:                                               ; preds = %84, %80
  %82 = phi i64 [ %91, %84 ], [ 0, %80 ]
  %83 = icmp slt i64 %82, 8
  br i1 %83, label %84, label %92

84:                                               ; preds = %81
  %85 = getelementptr inbounds float, ptr %72, i64 %82
  %86 = load float, ptr %85, align 4, !tbaa !4
  %87 = getelementptr inbounds float, ptr %10, i64 %82
  %88 = load float, ptr %87, align 4, !tbaa !4
  %89 = fmul float %86, %88
  %90 = getelementptr inbounds float, ptr %72, i64 %82
  store float %89, ptr %90, align 4, !tbaa !4
  %91 = add i64 %82, 1
  br label %81

92:                                               ; preds = %81
  %93 = getelementptr inbounds float, ptr %72, i32 7
  %94 = load float, ptr %93, align 4, !tbaa !4
  %95 = fpext float %94 to double
  %96 = load i64, ptr %42, align 4, !tbaa !1
  %97 = getelementptr inbounds double, ptr %36, i64 %96
  store double %95, ptr %97, align 8, !tbaa !6
  %98 = add i64 %96, 1
  store i64 %98, ptr %42, align 4, !tbaa !1
  %99 = fpext float %66 to double
  %100 = load i64, ptr %42, align 4, !tbaa !1
  %101 = getelementptr inbounds double, ptr %36, i64 %100
  store double %99, ptr %101, align 8, !tbaa !6
  %102 = add i64 %100, 1
  store i64 %102, ptr %42, align 4, !tbaa !1
  %103 = fpext float %64 to double
  %104 = load i64, ptr %42, align 4, !tbaa !1
  %105 = getelementptr inbounds double, ptr %36, i64 %104
  store double %103, ptr %105, align 8, !tbaa !6
  %106 = add i64 %104, 1
  store i64 %106, ptr %42, align 4, !tbaa !1
  %107 = fpext float %62 to double
  %108 = load i64, ptr %42, align 4, !tbaa !1
  %109 = getelementptr inbounds double, ptr %36, i64 %108
  store double %107, ptr %109, align 8, !tbaa !6
  %110 = add i64 %108, 1
  store i64 %110, ptr %42, align 4, !tbaa !1
  %111 = getelementptr inbounds float, ptr %1, i32 20
  %112 = load float, ptr %111, align 4, !tbaa !4
  %113 = getelementptr inbounds float, ptr %1, i32 19
  %114 = load float, ptr %113, align 4, !tbaa !4
  %115 = getelementptr inbounds float, ptr %1, i32 18
  %116 = load float, ptr %115, align 4, !tbaa !4
  %117 = getelementptr inbounds float, ptr %72, i32 6
  %118 = load float, ptr %117, align 4, !tbaa !4
  %119 = fpext float %118 to double
  %120 = load i64, ptr %42, align 4, !tbaa !1
  %121 = getelementptr inbounds double, ptr %36, i64 %120
  store double %119, ptr %121, align 8, !tbaa !6
  %122 = add i64 %120, 1
  store i64 %122, ptr %42, align 4, !tbaa !1
  %123 = fpext float %116 to double
  %124 = load i64, ptr %42, align 4, !tbaa !1
  %125 = getelementptr inbounds double, ptr %36, i64 %124
  store double %123, ptr %125, align 8, !tbaa !6
  %126 = add i64 %124, 1
  store i64 %126, ptr %42, align 4, !tbaa !1
  %127 = fpext float %114 to double
  %128 = load i64, ptr %42, align 4, !tbaa !1
  %129 = getelementptr inbounds double, ptr %36, i64 %128
  store double %127, ptr %129, align 8, !tbaa !6
  %130 = add i64 %128, 1
  store i64 %130, ptr %42, align 4, !tbaa !1
  %131 = fpext float %112 to double
  %132 = load i64, ptr %42, align 4, !tbaa !1
  %133 = getelementptr inbounds double, ptr %36, i64 %132
  store double %131, ptr %133, align 8, !tbaa !6
  %134 = add i64 %132, 1
  store i64 %134, ptr %42, align 4, !tbaa !1
  %135 = getelementptr inbounds float, ptr %1, i32 17
  %136 = load float, ptr %135, align 4, !tbaa !4
  %137 = getelementptr inbounds float, ptr %1, i32 16
  %138 = load float, ptr %137, align 4, !tbaa !4
  %139 = getelementptr inbounds float, ptr %1, i32 15
  %140 = load float, ptr %139, align 4, !tbaa !4
  %141 = getelementptr inbounds float, ptr %72, i32 5
  %142 = load float, ptr %141, align 4, !tbaa !4
  %143 = fpext float %142 to double
  %144 = load i64, ptr %42, align 4, !tbaa !1
  %145 = getelementptr inbounds double, ptr %36, i64 %144
  store double %143, ptr %145, align 8, !tbaa !6
  %146 = add i64 %144, 1
  store i64 %146, ptr %42, align 4, !tbaa !1
  %147 = fpext float %140 to double
  %148 = load i64, ptr %42, align 4, !tbaa !1
  %149 = getelementptr inbounds double, ptr %36, i64 %148
  store double %147, ptr %149, align 8, !tbaa !6
  %150 = add i64 %148, 1
  store i64 %150, ptr %42, align 4, !tbaa !1
  %151 = fpext float %138 to double
  %152 = load i64, ptr %42, align 4, !tbaa !1
  %153 = getelementptr inbounds double, ptr %36, i64 %152
  store double %151, ptr %153, align 8, !tbaa !6
  %154 = add i64 %152, 1
  store i64 %154, ptr %42, align 4, !tbaa !1
  %155 = fpext float %136 to double
  %156 = load i64, ptr %42, align 4, !tbaa !1
  %157 = getelementptr inbounds double, ptr %36, i64 %156
  store double %155, ptr %157, align 8, !tbaa !6
  %158 = add i64 %156, 1
  store i64 %158, ptr %42, align 4, !tbaa !1
  %159 = getelementptr inbounds float, ptr %1, i32 14
  %160 = load float, ptr %159, align 4, !tbaa !4
  %161 = getelementptr inbounds float, ptr %1, i32 13
  %162 = load float, ptr %161, align 4, !tbaa !4
  %163 = getelementptr inbounds float, ptr %1, i32 12
  %164 = load float, ptr %163, align 4, !tbaa !4
  %165 = getelementptr inbounds float, ptr %72, i32 4
  %166 = load float, ptr %165, align 4, !tbaa !4
  %167 = fpext float %166 to double
  %168 = load i64, ptr %42, align 4, !tbaa !1
  %169 = getelementptr inbounds double, ptr %36, i64 %168
  store double %167, ptr %169, align 8, !tbaa !6
  %170 = add i64 %168, 1
  store i64 %170, ptr %42, align 4, !tbaa !1
  %171 = fpext float %164 to double
  %172 = load i64, ptr %42, align 4, !tbaa !1
  %173 = getelementptr inbounds double, ptr %36, i64 %172
  store double %171, ptr %173, align 8, !tbaa !6
  %174 = add i64 %172, 1
  store i64 %174, ptr %42, align 4, !tbaa !1
  %175 = fpext float %162 to double
  %176 = load i64, ptr %42, align 4, !tbaa !1
  %177 = getelementptr inbounds double, ptr %36, i64 %176
  store double %175, ptr %177, align 8, !tbaa !6
  %178 = add i64 %176, 1
  store i64 %178, ptr %42, align 4, !tbaa !1
  %179 = fpext float %160 to double
  %180 = load i64, ptr %42, align 4, !tbaa !1
  %181 = getelementptr inbounds double, ptr %36, i64 %180
  store double %179, ptr %181, align 8, !tbaa !6
  %182 = add i64 %180, 1
  store i64 %182, ptr %42, align 4, !tbaa !1
  %183 = getelementptr inbounds float, ptr %1, i32 11
  %184 = load float, ptr %183, align 4, !tbaa !4
  %185 = getelementptr inbounds float, ptr %1, i32 10
  %186 = load float, ptr %185, align 4, !tbaa !4
  %187 = getelementptr inbounds float, ptr %1, i32 9
  %188 = load float, ptr %187, align 4, !tbaa !4
  %189 = getelementptr inbounds float, ptr %72, i32 3
  %190 = load float, ptr %189, align 4, !tbaa !4
  %191 = fpext float %190 to double
  %192 = load i64, ptr %42, align 4, !tbaa !1
  %193 = getelementptr inbounds double, ptr %36, i64 %192
  store double %191, ptr %193, align 8, !tbaa !6
  %194 = add i64 %192, 1
  store i64 %194, ptr %42, align 4, !tbaa !1
  %195 = fpext float %188 to double
  %196 = load i64, ptr %42, align 4, !tbaa !1
  %197 = getelementptr inbounds double, ptr %36, i64 %196
  store double %195, ptr %197, align 8, !tbaa !6
  %198 = add i64 %196, 1
  store i64 %198, ptr %42, align 4, !tbaa !1
  %199 = fpext float %186 to double
  %200 = load i64, ptr %42, align 4, !tbaa !1
  %201 = getelementptr inbounds double, ptr %36, i64 %200
  store double %199, ptr %201, align 8, !tbaa !6
  %202 = add i64 %200, 1
  store i64 %202, ptr %42, align 4, !tbaa !1
  %203 = fpext float %184 to double
  %204 = load i64, ptr %42, align 4, !tbaa !1
  %205 = getelementptr inbounds double, ptr %36, i64 %204
  store double %203, ptr %205, align 8, !tbaa !6
  %206 = add i64 %204, 1
  store i64 %206, ptr %42, align 4, !tbaa !1
  %207 = getelementptr inbounds float, ptr %1, i32 8
  %208 = load float, ptr %207, align 4, !tbaa !4
  %209 = getelementptr inbounds float, ptr %1, i32 7
  %210 = load float, ptr %209, align 4, !tbaa !4
  %211 = getelementptr inbounds float, ptr %1, i32 6
  %212 = load float, ptr %211, align 4, !tbaa !4
  %213 = getelementptr inbounds float, ptr %72, i32 2
  %214 = load float, ptr %213, align 4, !tbaa !4
  %215 = fpext float %214 to double
  %216 = load i64, ptr %42, align 4, !tbaa !1
  %217 = getelementptr inbounds double, ptr %36, i64 %216
  store double %215, ptr %217, align 8, !tbaa !6
  %218 = add i64 %216, 1
  store i64 %218, ptr %42, align 4, !tbaa !1
  %219 = fpext float %212 to double
  %220 = load i64, ptr %42, align 4, !tbaa !1
  %221 = getelementptr inbounds double, ptr %36, i64 %220
  store double %219, ptr %221, align 8, !tbaa !6
  %222 = add i64 %220, 1
  store i64 %222, ptr %42, align 4, !tbaa !1
  %223 = fpext float %210 to double
  %224 = load i64, ptr %42, align 4, !tbaa !1
  %225 = getelementptr inbounds double, ptr %36, i64 %224
  store double %223, ptr %225, align 8, !tbaa !6
  %226 = add i64 %224, 1
  store i64 %226, ptr %42, align 4, !tbaa !1
  %227 = fpext float %208 to double
  %228 = load i64, ptr %42, align 4, !tbaa !1
  %229 = getelementptr inbounds double, ptr %36, i64 %228
  store double %227, ptr %229, align 8, !tbaa !6
  %230 = add i64 %228, 1
  store i64 %230, ptr %42, align 4, !tbaa !1
  %231 = getelementptr inbounds float, ptr %1, i32 2
  %232 = load float, ptr %231, align 4, !tbaa !4
  %233 = getelementptr inbounds float, ptr %1, i32 1
  %234 = load float, ptr %233, align 4, !tbaa !4
  %235 = load float, ptr %1, align 4, !tbaa !4
  %236 = load float, ptr %72, align 4, !tbaa !4
  %237 = fpext float %236 to double
  %238 = load i64, ptr %42, align 4, !tbaa !1
  %239 = getelementptr inbounds double, ptr %36, i64 %238
  store double %237, ptr %239, align 8, !tbaa !6
  %240 = add i64 %238, 1
  store i64 %240, ptr %42, align 4, !tbaa !1
  %241 = fpext float %235 to double
  %242 = load i64, ptr %42, align 4, !tbaa !1
  %243 = getelementptr inbounds double, ptr %36, i64 %242
  store double %241, ptr %243, align 8, !tbaa !6
  %244 = add i64 %242, 1
  store i64 %244, ptr %42, align 4, !tbaa !1
  %245 = fpext float %234 to double
  %246 = load i64, ptr %42, align 4, !tbaa !1
  %247 = getelementptr inbounds double, ptr %36, i64 %246
  store double %245, ptr %247, align 8, !tbaa !6
  %248 = add i64 %246, 1
  store i64 %248, ptr %42, align 4, !tbaa !1
  %249 = fpext float %232 to double
  %250 = load i64, ptr %42, align 4, !tbaa !1
  %251 = getelementptr inbounds double, ptr %36, i64 %250
  store double %249, ptr %251, align 8, !tbaa !6
  %252 = add i64 %250, 1
  store i64 %252, ptr %42, align 4, !tbaa !1
  %253 = getelementptr inbounds float, ptr %1, i32 5
  %254 = load float, ptr %253, align 4, !tbaa !4
  %255 = getelementptr inbounds float, ptr %1, i32 4
  %256 = load float, ptr %255, align 4, !tbaa !4
  %257 = getelementptr inbounds float, ptr %1, i32 3
  %258 = load float, ptr %257, align 4, !tbaa !4
  %259 = getelementptr inbounds float, ptr %72, i32 1
  %260 = load float, ptr %259, align 4, !tbaa !4
  call void @_mlir_memref_to_llvm_free(ptr %67)
  %261 = fpext float %260 to double
  %262 = load i64, ptr %42, align 4, !tbaa !1
  %263 = getelementptr inbounds double, ptr %36, i64 %262
  store double %261, ptr %263, align 8, !tbaa !6
  %264 = add i64 %262, 1
  store i64 %264, ptr %42, align 4, !tbaa !1
  %265 = fpext float %258 to double
  %266 = load i64, ptr %42, align 4, !tbaa !1
  %267 = getelementptr inbounds double, ptr %36, i64 %266
  store double %265, ptr %267, align 8, !tbaa !6
  %268 = add i64 %266, 1
  store i64 %268, ptr %42, align 4, !tbaa !1
  %269 = fpext float %256 to double
  %270 = load i64, ptr %42, align 4, !tbaa !1
  %271 = getelementptr inbounds double, ptr %36, i64 %270
  store double %269, ptr %271, align 8, !tbaa !6
  %272 = add i64 %270, 1
  store i64 %272, ptr %42, align 4, !tbaa !1
  %273 = fpext float %254 to double
  %274 = load i64, ptr %42, align 4, !tbaa !1
  %275 = getelementptr inbounds double, ptr %36, i64 %274
  store double %273, ptr %275, align 8, !tbaa !6
  %276 = add i64 %274, 1
  store i64 %276, ptr %42, align 4, !tbaa !1
  %277 = fpext float %60 to double
  %278 = load i64, ptr %42, align 4, !tbaa !1
  %279 = getelementptr inbounds double, ptr %36, i64 %278
  store double %277, ptr %279, align 8, !tbaa !6
  %280 = add i64 %278, 1
  store i64 %280, ptr %42, align 4, !tbaa !1
  %281 = fpext float %58 to double
  %282 = load i64, ptr %42, align 4, !tbaa !1
  %283 = getelementptr inbounds double, ptr %36, i64 %282
  store double %281, ptr %283, align 8, !tbaa !6
  %284 = add i64 %282, 1
  store i64 %284, ptr %42, align 4, !tbaa !1
  %285 = fpext float %56 to double
  %286 = load i64, ptr %42, align 4, !tbaa !1
  %287 = getelementptr inbounds double, ptr %36, i64 %286
  store double %285, ptr %287, align 8, !tbaa !6
  %288 = add i64 %286, 1
  store i64 %288, ptr %42, align 4, !tbaa !1
  %289 = getelementptr inbounds float, ptr %1, i32 38
  %290 = load float, ptr %289, align 4, !tbaa !4
  %291 = getelementptr inbounds float, ptr %1, i32 37
  %292 = load float, ptr %291, align 4, !tbaa !4
  %293 = getelementptr inbounds float, ptr %1, i32 36
  %294 = load float, ptr %293, align 4, !tbaa !4
  %295 = fpext float %294 to double
  %296 = load i64, ptr %42, align 4, !tbaa !1
  %297 = getelementptr inbounds double, ptr %36, i64 %296
  store double %295, ptr %297, align 8, !tbaa !6
  %298 = add i64 %296, 1
  store i64 %298, ptr %42, align 4, !tbaa !1
  %299 = fpext float %292 to double
  %300 = load i64, ptr %42, align 4, !tbaa !1
  %301 = getelementptr inbounds double, ptr %36, i64 %300
  store double %299, ptr %301, align 8, !tbaa !6
  %302 = add i64 %300, 1
  store i64 %302, ptr %42, align 4, !tbaa !1
  %303 = fpext float %290 to double
  %304 = load i64, ptr %42, align 4, !tbaa !1
  %305 = getelementptr inbounds double, ptr %36, i64 %304
  store double %303, ptr %305, align 8, !tbaa !6
  %306 = add i64 %304, 1
  store i64 %306, ptr %42, align 4, !tbaa !1
  %307 = getelementptr inbounds float, ptr %1, i32 26
  %308 = load float, ptr %307, align 4, !tbaa !4
  %309 = getelementptr inbounds float, ptr %1, i32 25
  %310 = load float, ptr %309, align 4, !tbaa !4
  %311 = getelementptr inbounds float, ptr %1, i32 24
  %312 = load float, ptr %311, align 4, !tbaa !4
  %313 = fpext float %312 to double
  %314 = load i64, ptr %42, align 4, !tbaa !1
  %315 = getelementptr inbounds double, ptr %36, i64 %314
  store double %313, ptr %315, align 8, !tbaa !6
  %316 = add i64 %314, 1
  store i64 %316, ptr %42, align 4, !tbaa !1
  %317 = fpext float %310 to double
  %318 = load i64, ptr %42, align 4, !tbaa !1
  %319 = getelementptr inbounds double, ptr %36, i64 %318
  store double %317, ptr %319, align 8, !tbaa !6
  %320 = add i64 %318, 1
  store i64 %320, ptr %42, align 4, !tbaa !1
  %321 = fpext float %308 to double
  %322 = load i64, ptr %42, align 4, !tbaa !1
  %323 = getelementptr inbounds double, ptr %36, i64 %322
  store double %321, ptr %323, align 8, !tbaa !6
  %324 = add i64 %322, 1
  store i64 %324, ptr %42, align 4, !tbaa !1
  %325 = getelementptr inbounds float, ptr %1, i32 32
  %326 = load float, ptr %325, align 4, !tbaa !4
  %327 = getelementptr inbounds float, ptr %1, i32 31
  %328 = load float, ptr %327, align 4, !tbaa !4
  %329 = getelementptr inbounds float, ptr %1, i32 30
  %330 = load float, ptr %329, align 4, !tbaa !4
  %331 = fpext float %330 to double
  %332 = load i64, ptr %42, align 4, !tbaa !1
  %333 = getelementptr inbounds double, ptr %36, i64 %332
  store double %331, ptr %333, align 8, !tbaa !6
  %334 = add i64 %332, 1
  store i64 %334, ptr %42, align 4, !tbaa !1
  %335 = fpext float %328 to double
  %336 = load i64, ptr %42, align 4, !tbaa !1
  %337 = getelementptr inbounds double, ptr %36, i64 %336
  store double %335, ptr %337, align 8, !tbaa !6
  %338 = add i64 %336, 1
  store i64 %338, ptr %42, align 4, !tbaa !1
  %339 = fpext float %326 to double
  %340 = load i64, ptr %42, align 4, !tbaa !1
  %341 = getelementptr inbounds double, ptr %36, i64 %340
  store double %339, ptr %341, align 8, !tbaa !6
  %342 = add i64 %340, 1
  store i64 %342, ptr %42, align 4, !tbaa !1
  %343 = fpext float %54 to double
  %344 = load i64, ptr %42, align 4, !tbaa !1
  %345 = getelementptr inbounds double, ptr %36, i64 %344
  store double %343, ptr %345, align 8, !tbaa !6
  %346 = add i64 %344, 1
  store i64 %346, ptr %42, align 4, !tbaa !1
  %347 = fpext float %52 to double
  %348 = load i64, ptr %42, align 4, !tbaa !1
  %349 = getelementptr inbounds double, ptr %36, i64 %348
  store double %347, ptr %349, align 8, !tbaa !6
  %350 = add i64 %348, 1
  store i64 %350, ptr %42, align 4, !tbaa !1
  %351 = fpext float %50 to double
  %352 = load i64, ptr %42, align 4, !tbaa !1
  %353 = getelementptr inbounds double, ptr %36, i64 %352
  store double %351, ptr %353, align 8, !tbaa !6
  %354 = add i64 %352, 1
  store i64 %354, ptr %42, align 4, !tbaa !1
  %355 = getelementptr inbounds float, ptr %1, i32 59
  %356 = load float, ptr %355, align 4, !tbaa !4
  %357 = getelementptr inbounds float, ptr %1, i32 58
  %358 = load float, ptr %357, align 4, !tbaa !4
  %359 = getelementptr inbounds float, ptr %1, i32 57
  %360 = load float, ptr %359, align 4, !tbaa !4
  %361 = getelementptr inbounds float, ptr %1, i32 41
  %362 = load float, ptr %361, align 4, !tbaa !4
  %363 = getelementptr inbounds float, ptr %1, i32 40
  %364 = load float, ptr %363, align 4, !tbaa !4
  %365 = getelementptr inbounds float, ptr %1, i32 39
  %366 = load float, ptr %365, align 4, !tbaa !4
  %367 = fpext float %366 to double
  %368 = load i64, ptr %42, align 4, !tbaa !1
  %369 = getelementptr inbounds double, ptr %36, i64 %368
  store double %367, ptr %369, align 8, !tbaa !6
  %370 = add i64 %368, 1
  store i64 %370, ptr %42, align 4, !tbaa !1
  %371 = fpext float %364 to double
  %372 = load i64, ptr %42, align 4, !tbaa !1
  %373 = getelementptr inbounds double, ptr %36, i64 %372
  store double %371, ptr %373, align 8, !tbaa !6
  %374 = add i64 %372, 1
  store i64 %374, ptr %42, align 4, !tbaa !1
  %375 = fpext float %362 to double
  %376 = load i64, ptr %42, align 4, !tbaa !1
  %377 = getelementptr inbounds double, ptr %36, i64 %376
  store double %375, ptr %377, align 8, !tbaa !6
  %378 = add i64 %376, 1
  store i64 %378, ptr %42, align 4, !tbaa !1
  %379 = getelementptr inbounds float, ptr %1, i32 29
  %380 = load float, ptr %379, align 4, !tbaa !4
  %381 = getelementptr inbounds float, ptr %1, i32 28
  %382 = load float, ptr %381, align 4, !tbaa !4
  %383 = getelementptr inbounds float, ptr %1, i32 27
  %384 = load float, ptr %383, align 4, !tbaa !4
  %385 = fpext float %384 to double
  %386 = load i64, ptr %42, align 4, !tbaa !1
  %387 = getelementptr inbounds double, ptr %36, i64 %386
  store double %385, ptr %387, align 8, !tbaa !6
  %388 = add i64 %386, 1
  store i64 %388, ptr %42, align 4, !tbaa !1
  %389 = fpext float %382 to double
  %390 = load i64, ptr %42, align 4, !tbaa !1
  %391 = getelementptr inbounds double, ptr %36, i64 %390
  store double %389, ptr %391, align 8, !tbaa !6
  %392 = add i64 %390, 1
  store i64 %392, ptr %42, align 4, !tbaa !1
  %393 = fpext float %380 to double
  %394 = load i64, ptr %42, align 4, !tbaa !1
  %395 = getelementptr inbounds double, ptr %36, i64 %394
  store double %393, ptr %395, align 8, !tbaa !6
  %396 = add i64 %394, 1
  store i64 %396, ptr %42, align 4, !tbaa !1
  %397 = getelementptr inbounds float, ptr %1, i32 35
  %398 = load float, ptr %397, align 4, !tbaa !4
  %399 = getelementptr inbounds float, ptr %1, i32 34
  %400 = load float, ptr %399, align 4, !tbaa !4
  %401 = getelementptr inbounds float, ptr %1, i32 33
  %402 = load float, ptr %401, align 4, !tbaa !4
  %403 = fpext float %402 to double
  %404 = load i64, ptr %42, align 4, !tbaa !1
  %405 = getelementptr inbounds double, ptr %36, i64 %404
  store double %403, ptr %405, align 8, !tbaa !6
  %406 = add i64 %404, 1
  store i64 %406, ptr %42, align 4, !tbaa !1
  %407 = fpext float %400 to double
  %408 = load i64, ptr %42, align 4, !tbaa !1
  %409 = getelementptr inbounds double, ptr %36, i64 %408
  store double %407, ptr %409, align 8, !tbaa !6
  %410 = add i64 %408, 1
  store i64 %410, ptr %42, align 4, !tbaa !1
  %411 = fpext float %398 to double
  %412 = load i64, ptr %42, align 4, !tbaa !1
  %413 = getelementptr inbounds double, ptr %36, i64 %412
  store double %411, ptr %413, align 8, !tbaa !6
  %414 = add i64 %412, 1
  store i64 %414, ptr %42, align 4, !tbaa !1
  %415 = fpext float %360 to double
  %416 = load i64, ptr %42, align 4, !tbaa !1
  %417 = getelementptr inbounds double, ptr %36, i64 %416
  store double %415, ptr %417, align 8, !tbaa !6
  %418 = add i64 %416, 1
  store i64 %418, ptr %42, align 4, !tbaa !1
  %419 = fpext float %358 to double
  %420 = load i64, ptr %42, align 4, !tbaa !1
  %421 = getelementptr inbounds double, ptr %36, i64 %420
  store double %419, ptr %421, align 8, !tbaa !6
  %422 = add i64 %420, 1
  store i64 %422, ptr %42, align 4, !tbaa !1
  %423 = fpext float %356 to double
  %424 = load i64, ptr %42, align 4, !tbaa !1
  %425 = getelementptr inbounds double, ptr %36, i64 %424
  store double %423, ptr %425, align 8, !tbaa !6
  %426 = add i64 %424, 1
  store i64 %426, ptr %42, align 4, !tbaa !1
  %427 = getelementptr inbounds float, ptr %1, i32 56
  %428 = load float, ptr %427, align 4, !tbaa !4
  %429 = getelementptr inbounds float, ptr %1, i32 55
  %430 = load float, ptr %429, align 4, !tbaa !4
  %431 = getelementptr inbounds float, ptr %1, i32 54
  %432 = load float, ptr %431, align 4, !tbaa !4
  %433 = fpext float %432 to double
  %434 = load i64, ptr %42, align 4, !tbaa !1
  %435 = getelementptr inbounds double, ptr %36, i64 %434
  store double %433, ptr %435, align 8, !tbaa !6
  %436 = add i64 %434, 1
  store i64 %436, ptr %42, align 4, !tbaa !1
  %437 = fpext float %430 to double
  %438 = load i64, ptr %42, align 4, !tbaa !1
  %439 = getelementptr inbounds double, ptr %36, i64 %438
  store double %437, ptr %439, align 8, !tbaa !6
  %440 = add i64 %438, 1
  store i64 %440, ptr %42, align 4, !tbaa !1
  %441 = fpext float %428 to double
  %442 = load i64, ptr %42, align 4, !tbaa !1
  %443 = getelementptr inbounds double, ptr %36, i64 %442
  store double %441, ptr %443, align 8, !tbaa !6
  %444 = add i64 %442, 1
  store i64 %444, ptr %42, align 4, !tbaa !1
  %445 = getelementptr inbounds float, ptr %1, i32 65
  %446 = load float, ptr %445, align 4, !tbaa !4
  %447 = getelementptr inbounds float, ptr %1, i32 64
  %448 = load float, ptr %447, align 4, !tbaa !4
  %449 = getelementptr inbounds float, ptr %1, i32 63
  %450 = load float, ptr %449, align 4, !tbaa !4
  %451 = getelementptr inbounds float, ptr %1, i32 47
  %452 = load float, ptr %451, align 4, !tbaa !4
  %453 = getelementptr inbounds float, ptr %1, i32 46
  %454 = load float, ptr %453, align 4, !tbaa !4
  %455 = getelementptr inbounds float, ptr %1, i32 45
  %456 = load float, ptr %455, align 4, !tbaa !4
  %457 = fpext float %456 to double
  %458 = load i64, ptr %42, align 4, !tbaa !1
  %459 = getelementptr inbounds double, ptr %36, i64 %458
  store double %457, ptr %459, align 8, !tbaa !6
  %460 = add i64 %458, 1
  store i64 %460, ptr %42, align 4, !tbaa !1
  %461 = fpext float %454 to double
  %462 = load i64, ptr %42, align 4, !tbaa !1
  %463 = getelementptr inbounds double, ptr %36, i64 %462
  store double %461, ptr %463, align 8, !tbaa !6
  %464 = add i64 %462, 1
  store i64 %464, ptr %42, align 4, !tbaa !1
  %465 = fpext float %452 to double
  %466 = load i64, ptr %42, align 4, !tbaa !1
  %467 = getelementptr inbounds double, ptr %36, i64 %466
  store double %465, ptr %467, align 8, !tbaa !6
  %468 = add i64 %466, 1
  store i64 %468, ptr %42, align 4, !tbaa !1
  %469 = fpext float %450 to double
  %470 = load i64, ptr %42, align 4, !tbaa !1
  %471 = getelementptr inbounds double, ptr %36, i64 %470
  store double %469, ptr %471, align 8, !tbaa !6
  %472 = add i64 %470, 1
  store i64 %472, ptr %42, align 4, !tbaa !1
  %473 = fpext float %448 to double
  %474 = load i64, ptr %42, align 4, !tbaa !1
  %475 = getelementptr inbounds double, ptr %36, i64 %474
  store double %473, ptr %475, align 8, !tbaa !6
  %476 = add i64 %474, 1
  store i64 %476, ptr %42, align 4, !tbaa !1
  %477 = fpext float %446 to double
  %478 = load i64, ptr %42, align 4, !tbaa !1
  %479 = getelementptr inbounds double, ptr %36, i64 %478
  store double %477, ptr %479, align 8, !tbaa !6
  %480 = add i64 %478, 1
  store i64 %480, ptr %42, align 4, !tbaa !1
  %481 = fpext float %48 to double
  %482 = load i64, ptr %42, align 4, !tbaa !1
  %483 = getelementptr inbounds double, ptr %36, i64 %482
  store double %481, ptr %483, align 8, !tbaa !6
  %484 = add i64 %482, 1
  store i64 %484, ptr %42, align 4, !tbaa !1
  %485 = fpext float %46 to double
  %486 = load i64, ptr %42, align 4, !tbaa !1
  %487 = getelementptr inbounds double, ptr %36, i64 %486
  store double %485, ptr %487, align 8, !tbaa !6
  %488 = add i64 %486, 1
  store i64 %488, ptr %42, align 4, !tbaa !1
  %489 = fpext float %44 to double
  %490 = load i64, ptr %42, align 4, !tbaa !1
  %491 = getelementptr inbounds double, ptr %36, i64 %490
  store double %489, ptr %491, align 8, !tbaa !6
  %492 = add i64 %490, 1
  store i64 %492, ptr %42, align 4, !tbaa !1
  %493 = getelementptr inbounds float, ptr %1, i32 86
  %494 = load float, ptr %493, align 4, !tbaa !4
  %495 = getelementptr inbounds float, ptr %1, i32 85
  %496 = load float, ptr %495, align 4, !tbaa !4
  %497 = getelementptr inbounds float, ptr %1, i32 84
  %498 = load float, ptr %497, align 4, !tbaa !4
  %499 = getelementptr inbounds float, ptr %1, i32 71
  %500 = load float, ptr %499, align 4, !tbaa !4
  %501 = getelementptr inbounds float, ptr %1, i32 70
  %502 = load float, ptr %501, align 4, !tbaa !4
  %503 = getelementptr inbounds float, ptr %1, i32 69
  %504 = load float, ptr %503, align 4, !tbaa !4
  %505 = fpext float %504 to double
  %506 = load i64, ptr %42, align 4, !tbaa !1
  %507 = getelementptr inbounds double, ptr %36, i64 %506
  store double %505, ptr %507, align 8, !tbaa !6
  %508 = add i64 %506, 1
  store i64 %508, ptr %42, align 4, !tbaa !1
  %509 = fpext float %502 to double
  %510 = load i64, ptr %42, align 4, !tbaa !1
  %511 = getelementptr inbounds double, ptr %36, i64 %510
  store double %509, ptr %511, align 8, !tbaa !6
  %512 = add i64 %510, 1
  store i64 %512, ptr %42, align 4, !tbaa !1
  %513 = fpext float %500 to double
  %514 = load i64, ptr %42, align 4, !tbaa !1
  %515 = getelementptr inbounds double, ptr %36, i64 %514
  store double %513, ptr %515, align 8, !tbaa !6
  %516 = add i64 %514, 1
  store i64 %516, ptr %42, align 4, !tbaa !1
  %517 = getelementptr inbounds float, ptr %1, i32 53
  %518 = load float, ptr %517, align 4, !tbaa !4
  %519 = getelementptr inbounds float, ptr %1, i32 52
  %520 = load float, ptr %519, align 4, !tbaa !4
  %521 = getelementptr inbounds float, ptr %1, i32 51
  %522 = load float, ptr %521, align 4, !tbaa !4
  %523 = fpext float %522 to double
  %524 = load i64, ptr %42, align 4, !tbaa !1
  %525 = getelementptr inbounds double, ptr %36, i64 %524
  store double %523, ptr %525, align 8, !tbaa !6
  %526 = add i64 %524, 1
  store i64 %526, ptr %42, align 4, !tbaa !1
  %527 = fpext float %520 to double
  %528 = load i64, ptr %42, align 4, !tbaa !1
  %529 = getelementptr inbounds double, ptr %36, i64 %528
  store double %527, ptr %529, align 8, !tbaa !6
  %530 = add i64 %528, 1
  store i64 %530, ptr %42, align 4, !tbaa !1
  %531 = fpext float %518 to double
  %532 = load i64, ptr %42, align 4, !tbaa !1
  %533 = getelementptr inbounds double, ptr %36, i64 %532
  store double %531, ptr %533, align 8, !tbaa !6
  %534 = add i64 %532, 1
  store i64 %534, ptr %42, align 4, !tbaa !1
  %535 = getelementptr inbounds float, ptr %1, i32 62
  %536 = load float, ptr %535, align 4, !tbaa !4
  %537 = getelementptr inbounds float, ptr %1, i32 61
  %538 = load float, ptr %537, align 4, !tbaa !4
  %539 = getelementptr inbounds float, ptr %1, i32 60
  %540 = load float, ptr %539, align 4, !tbaa !4
  %541 = fpext float %540 to double
  %542 = load i64, ptr %42, align 4, !tbaa !1
  %543 = getelementptr inbounds double, ptr %36, i64 %542
  store double %541, ptr %543, align 8, !tbaa !6
  %544 = add i64 %542, 1
  store i64 %544, ptr %42, align 4, !tbaa !1
  %545 = fpext float %538 to double
  %546 = load i64, ptr %42, align 4, !tbaa !1
  %547 = getelementptr inbounds double, ptr %36, i64 %546
  store double %545, ptr %547, align 8, !tbaa !6
  %548 = add i64 %546, 1
  store i64 %548, ptr %42, align 4, !tbaa !1
  %549 = fpext float %536 to double
  %550 = load i64, ptr %42, align 4, !tbaa !1
  %551 = getelementptr inbounds double, ptr %36, i64 %550
  store double %549, ptr %551, align 8, !tbaa !6
  %552 = add i64 %550, 1
  store i64 %552, ptr %42, align 4, !tbaa !1
  %553 = fpext float %498 to double
  %554 = load i64, ptr %42, align 4, !tbaa !1
  %555 = getelementptr inbounds double, ptr %36, i64 %554
  store double %553, ptr %555, align 8, !tbaa !6
  %556 = add i64 %554, 1
  store i64 %556, ptr %42, align 4, !tbaa !1
  %557 = fpext float %496 to double
  %558 = load i64, ptr %42, align 4, !tbaa !1
  %559 = getelementptr inbounds double, ptr %36, i64 %558
  store double %557, ptr %559, align 8, !tbaa !6
  %560 = add i64 %558, 1
  store i64 %560, ptr %42, align 4, !tbaa !1
  %561 = fpext float %494 to double
  %562 = load i64, ptr %42, align 4, !tbaa !1
  %563 = getelementptr inbounds double, ptr %36, i64 %562
  store double %561, ptr %563, align 8, !tbaa !6
  %564 = add i64 %562, 1
  store i64 %564, ptr %42, align 4, !tbaa !1
  %565 = getelementptr inbounds float, ptr %1, i32 77
  %566 = load float, ptr %565, align 4, !tbaa !4
  %567 = getelementptr inbounds float, ptr %1, i32 76
  %568 = load float, ptr %567, align 4, !tbaa !4
  %569 = getelementptr inbounds float, ptr %1, i32 75
  %570 = load float, ptr %569, align 4, !tbaa !4
  %571 = getelementptr inbounds float, ptr %1, i32 68
  %572 = load float, ptr %571, align 4, !tbaa !4
  %573 = getelementptr inbounds float, ptr %1, i32 67
  %574 = load float, ptr %573, align 4, !tbaa !4
  %575 = getelementptr inbounds float, ptr %1, i32 66
  %576 = load float, ptr %575, align 4, !tbaa !4
  %577 = fpext float %576 to double
  %578 = load i64, ptr %42, align 4, !tbaa !1
  %579 = getelementptr inbounds double, ptr %36, i64 %578
  store double %577, ptr %579, align 8, !tbaa !6
  %580 = add i64 %578, 1
  store i64 %580, ptr %42, align 4, !tbaa !1
  %581 = fpext float %574 to double
  %582 = load i64, ptr %42, align 4, !tbaa !1
  %583 = getelementptr inbounds double, ptr %36, i64 %582
  store double %581, ptr %583, align 8, !tbaa !6
  %584 = add i64 %582, 1
  store i64 %584, ptr %42, align 4, !tbaa !1
  %585 = fpext float %572 to double
  %586 = load i64, ptr %42, align 4, !tbaa !1
  %587 = getelementptr inbounds double, ptr %36, i64 %586
  store double %585, ptr %587, align 8, !tbaa !6
  %588 = add i64 %586, 1
  store i64 %588, ptr %42, align 4, !tbaa !1
  %589 = fpext float %570 to double
  %590 = load i64, ptr %42, align 4, !tbaa !1
  %591 = getelementptr inbounds double, ptr %36, i64 %590
  store double %589, ptr %591, align 8, !tbaa !6
  %592 = add i64 %590, 1
  store i64 %592, ptr %42, align 4, !tbaa !1
  %593 = fpext float %568 to double
  %594 = load i64, ptr %42, align 4, !tbaa !1
  %595 = getelementptr inbounds double, ptr %36, i64 %594
  store double %593, ptr %595, align 8, !tbaa !6
  %596 = add i64 %594, 1
  store i64 %596, ptr %42, align 4, !tbaa !1
  %597 = fpext float %566 to double
  %598 = load i64, ptr %42, align 4, !tbaa !1
  %599 = getelementptr inbounds double, ptr %36, i64 %598
  store double %597, ptr %599, align 8, !tbaa !6
  %600 = add i64 %598, 1
  store i64 %600, ptr %42, align 4, !tbaa !1
  %601 = getelementptr inbounds float, ptr %1, i32 89
  %602 = load float, ptr %601, align 4, !tbaa !4
  %603 = getelementptr inbounds float, ptr %1, i32 88
  %604 = load float, ptr %603, align 4, !tbaa !4
  %605 = getelementptr inbounds float, ptr %1, i32 87
  %606 = load float, ptr %605, align 4, !tbaa !4
  %607 = fpext float %606 to double
  %608 = load i64, ptr %42, align 4, !tbaa !1
  %609 = getelementptr inbounds double, ptr %36, i64 %608
  store double %607, ptr %609, align 8, !tbaa !6
  %610 = add i64 %608, 1
  store i64 %610, ptr %42, align 4, !tbaa !1
  %611 = fpext float %604 to double
  %612 = load i64, ptr %42, align 4, !tbaa !1
  %613 = getelementptr inbounds double, ptr %36, i64 %612
  store double %611, ptr %613, align 8, !tbaa !6
  %614 = add i64 %612, 1
  store i64 %614, ptr %42, align 4, !tbaa !1
  %615 = fpext float %602 to double
  %616 = load i64, ptr %42, align 4, !tbaa !1
  %617 = getelementptr inbounds double, ptr %36, i64 %616
  store double %615, ptr %617, align 8, !tbaa !6
  %618 = add i64 %616, 1
  store i64 %618, ptr %42, align 4, !tbaa !1
  %619 = getelementptr inbounds float, ptr %1, i32 80
  %620 = load float, ptr %619, align 4, !tbaa !4
  %621 = getelementptr inbounds float, ptr %1, i32 79
  %622 = load float, ptr %621, align 4, !tbaa !4
  %623 = getelementptr inbounds float, ptr %1, i32 78
  %624 = load float, ptr %623, align 4, !tbaa !4
  %625 = fpext float %624 to double
  %626 = load i64, ptr %42, align 4, !tbaa !1
  %627 = getelementptr inbounds double, ptr %36, i64 %626
  store double %625, ptr %627, align 8, !tbaa !6
  %628 = add i64 %626, 1
  store i64 %628, ptr %42, align 4, !tbaa !1
  %629 = fpext float %622 to double
  %630 = load i64, ptr %42, align 4, !tbaa !1
  %631 = getelementptr inbounds double, ptr %36, i64 %630
  store double %629, ptr %631, align 8, !tbaa !6
  %632 = add i64 %630, 1
  store i64 %632, ptr %42, align 4, !tbaa !1
  %633 = fpext float %620 to double
  %634 = load i64, ptr %42, align 4, !tbaa !1
  %635 = getelementptr inbounds double, ptr %36, i64 %634
  store double %633, ptr %635, align 8, !tbaa !6
  %636 = add i64 %634, 1
  store i64 %636, ptr %42, align 4, !tbaa !1
  %637 = getelementptr inbounds float, ptr %1, i32 92
  %638 = load float, ptr %637, align 4, !tbaa !4
  %639 = getelementptr inbounds float, ptr %1, i32 91
  %640 = load float, ptr %639, align 4, !tbaa !4
  %641 = getelementptr inbounds float, ptr %1, i32 90
  %642 = load float, ptr %641, align 4, !tbaa !4
  %643 = fpext float %642 to double
  %644 = load i64, ptr %42, align 4, !tbaa !1
  %645 = getelementptr inbounds double, ptr %36, i64 %644
  store double %643, ptr %645, align 8, !tbaa !6
  %646 = add i64 %644, 1
  store i64 %646, ptr %42, align 4, !tbaa !1
  %647 = fpext float %640 to double
  %648 = load i64, ptr %42, align 4, !tbaa !1
  %649 = getelementptr inbounds double, ptr %36, i64 %648
  store double %647, ptr %649, align 8, !tbaa !6
  %650 = add i64 %648, 1
  store i64 %650, ptr %42, align 4, !tbaa !1
  %651 = fpext float %638 to double
  %652 = load i64, ptr %42, align 4, !tbaa !1
  %653 = getelementptr inbounds double, ptr %36, i64 %652
  store double %651, ptr %653, align 8, !tbaa !6
  %654 = add i64 %652, 1
  store i64 %654, ptr %42, align 4, !tbaa !1
  %655 = getelementptr inbounds float, ptr %1, i32 83
  %656 = load float, ptr %655, align 4, !tbaa !4
  %657 = getelementptr inbounds float, ptr %1, i32 82
  %658 = load float, ptr %657, align 4, !tbaa !4
  %659 = getelementptr inbounds float, ptr %1, i32 81
  %660 = load float, ptr %659, align 4, !tbaa !4
  %661 = fpext float %660 to double
  %662 = load i64, ptr %42, align 4, !tbaa !1
  %663 = getelementptr inbounds double, ptr %36, i64 %662
  store double %661, ptr %663, align 8, !tbaa !6
  %664 = add i64 %662, 1
  store i64 %664, ptr %42, align 4, !tbaa !1
  %665 = fpext float %658 to double
  %666 = load i64, ptr %42, align 4, !tbaa !1
  %667 = getelementptr inbounds double, ptr %36, i64 %666
  store double %665, ptr %667, align 8, !tbaa !6
  %668 = add i64 %666, 1
  store i64 %668, ptr %42, align 4, !tbaa !1
  %669 = fpext float %656 to double
  %670 = load i64, ptr %42, align 4, !tbaa !1
  %671 = getelementptr inbounds double, ptr %36, i64 %670
  store double %669, ptr %671, align 8, !tbaa !6
  %672 = add i64 %670, 1
  store i64 %672, ptr %42, align 4, !tbaa !1
  %673 = getelementptr inbounds float, ptr %1, i32 95
  %674 = load float, ptr %673, align 4, !tbaa !4
  %675 = getelementptr inbounds float, ptr %1, i32 94
  %676 = load float, ptr %675, align 4, !tbaa !4
  %677 = getelementptr inbounds float, ptr %1, i32 93
  %678 = load float, ptr %677, align 4, !tbaa !4
  %679 = fpext float %678 to double
  %680 = load i64, ptr %42, align 4, !tbaa !1
  %681 = getelementptr inbounds double, ptr %36, i64 %680
  store double %679, ptr %681, align 8, !tbaa !6
  %682 = add i64 %680, 1
  store i64 %682, ptr %42, align 4, !tbaa !1
  %683 = fpext float %676 to double
  %684 = load i64, ptr %42, align 4, !tbaa !1
  %685 = getelementptr inbounds double, ptr %36, i64 %684
  store double %683, ptr %685, align 8, !tbaa !6
  %686 = add i64 %684, 1
  store i64 %686, ptr %42, align 4, !tbaa !1
  %687 = fpext float %674 to double
  %688 = load i64, ptr %42, align 4, !tbaa !1
  %689 = getelementptr inbounds double, ptr %36, i64 %688
  store double %687, ptr %689, align 8, !tbaa !6
  %690 = add i64 %688, 1
  store i64 %690, ptr %42, align 4, !tbaa !1
  store { ptr, ptr, i64, [3 x i64], [3 x i64] } %29, ptr %33, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, ptr %32, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %41, ptr %31, align 8
  %691 = call ptr @_mlir_memref_to_llvm_alloc(i64 8)
  %692 = insertvalue { ptr, ptr, i64 } poison, ptr %691, 0
  %693 = insertvalue { ptr, ptr, i64 } %692, ptr %691, 1
  %694 = insertvalue { ptr, ptr, i64 } %693, i64 0, 2
  store { ptr, ptr, i64 } %694, ptr %30, align 8
  call void @qnode_forward_0.quantum(ptr %33, ptr %32, ptr %31, ptr %30)
  ret { ptr, ptr, i64 } %694
}

define internal void @_sample_loss.cloned(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, ptr %9, ptr %10, i64 %11, ptr %12, ptr %13, i64 %14, ptr %15, ptr %16, i64 %17, i64 %18, i64 %19, ptr %20, ptr %21, i64 %22, ptr %23, ptr %24, i64 %25, ptr %26, ptr %27, i64 %28) {
  %30 = call i64 @qnode_forward_0.pcount(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, ptr %15, ptr %16, i64 %17, i64 %18, i64 %19)
  %31 = load float, ptr %24, align 4, !tbaa !4
  %32 = load float, ptr %21, align 4, !tbaa !4
  %33 = load float, ptr %10, align 4, !tbaa !4
  %34 = load float, ptr %13, align 4, !tbaa !4
  %35 = call { ptr, ptr, i64 } @qnode_forward_0.preprocess(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, ptr %15, ptr %16, i64 %17, i64 %18, i64 %19, i64 %30)
  %36 = extractvalue { ptr, ptr, i64 } %35, 1
  %37 = load double, ptr %36, align 8, !tbaa !6
  %38 = fpext float %34 to double
  %39 = fmul double %38, %37
  %40 = fpext float %33 to double
  %41 = fadd double %39, %40
  %42 = fpext float %32 to double
  %43 = fpext float %31 to double
  %44 = fcmp ugt double %41, 0.000000e+00
  %45 = select i1 %44, double %41, double 0.000000e+00
  %46 = select i1 false, double 0.000000e+00, double %45
  %47 = fcmp une double %41, %41
  %48 = call double @llvm.fabs.f64(double %41)
  %49 = fneg double %48
  %50 = call double @llvm.exp.f64(double %49)
  %51 = fadd double 1.000000e+00, %50
  %52 = call double @llvm.log.f64(double %51)
  %53 = fadd double %46, %52
  %54 = select i1 %47, double %41, double %53
  %55 = fmul double %42, %41
  %56 = fsub double %54, %55
  %57 = fmul double %43, %56
  %58 = call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %59 = ptrtoint ptr %58 to i64
  %60 = add i64 %59, 63
  %61 = urem i64 %60, 64
  %62 = sub i64 %60, %61
  %63 = inttoptr i64 %62 to ptr
  store double %57, ptr %63, align 8, !tbaa !6
  %64 = load double, ptr %63, align 8, !tbaa !6
  store double %64, ptr %27, align 8, !tbaa !6
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

define internal double @softplus_2.detensorized(double %0) {
  %2 = fcmp ugt double %0, 0.000000e+00
  %3 = select i1 %2, double %0, double 0.000000e+00
  %4 = select i1 false, double 0.000000e+00, double %3
  %5 = fcmp une double %0, %0
  %6 = call double @llvm.fabs.f64(double %0)
  %7 = fneg double %6
  %8 = call double @llvm.exp.f64(double %7)
  %9 = fadd double 1.000000e+00, %8
  %10 = call double @llvm.log.f64(double %9)
  %11 = fadd double %4, %10
  %12 = select i1 %5, double %0, double %11
  ret double %12
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fabs.f64(double) #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.exp.f64(double) #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.log.f64(double) #2

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #3

attributes #0 = { noinline }
attributes #1 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { nocallback nofree nounwind willreturn memory(argmem: write) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{!2, !2, i64 0}
!2 = !{!"int", !3, i64 0}
!3 = !{!"Catalyst TBAA"}
!4 = !{!5, !5, i64 0}
!5 = !{!"float", !3, i64 0}
!6 = !{!7, !7, i64 0}
!7 = !{!"double", !3, i64 0}
