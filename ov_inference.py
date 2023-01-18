import openvino.runtime as ov
import numpy as np


core = ov.Core()

compiled_model = core.compile_model("/home/ethan/Downloads/PaddleOCRv3/en_ppocr_mobile_v2.0_table_structure_infer/table.onnx", "CPU")
infer_request = compiled_model.create_infer_request()

data = np.ones((1,3,488,488))
data  = data.astype('float32')

result_infer = compiled_model([data])

result_infer