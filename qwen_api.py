from flask import Flask, request, jsonify
import datetime
import os
# import tensorflow as tf
import pdb
# from tqdm import tqdm
# from flask import Flask, request, jsonify
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import json
from flask import Flask
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
model_name = '/data/luohh/acl23/Qwen-VL/yingji_7b_sft_3epoch'
# tokenizer = AutoTokenizer.from_pretrained(model_name,local_files_only=True, trust_remote_code=True)
# model = AutoPeftModelForCausalLM.from_pretrained(
#     model_name, # path to the output directory
#     device_map="cuda",
#     trust_remote_code=True
# ).eval()
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name,local_files_only=True, trust_remote_code=True)

app = Flask(__name__)
@app.route('/', methods=['POST'])
def predict():
    query = request.get_json()
#     query = query['instruction']
#     if query == '火灾后，如何为受灾群众以及救灾消防官兵提供心理辅导':
#         response = '火灾后，为受灾群众和救灾消防官兵提供心理辅导是非常重要的。首先，需要建立心理支持团队，包括心理医生、心理治疗师、心理咨询师等专业人员，为受灾群众和消防官兵提供心理支持和疏导。其次，可以通过心理教育、心理疏导、心理治疗等方式，帮助他们理解和应对心理问题，减轻心理压力和焦虑。同时，还可以组织心理康复活动，如心理支持小组、心理康复训练等，促进受灾群众和消防官兵的心理康复和恢复。最重要的是，要关注他们的心理需求和变化，及时调整和改进心理辅导方案，确保他们得到全面的心理支持和帮助。'
#     elif query == "请简述高空作业安全管理制度。":
#         response = '高空作业安全管理制度包括：1. 高空作业人员必须经过专业培训，持证上岗；2. 高空作业前必须进行安全技术交底，确保作业人员了解安全操作规程；3. 高空作业人员必须佩戴安全带，确保安全带牢固可靠；4. 高空作业现场必须设置安全警示标志，确保作业人员注意安全；5. 高空作业现场必须配备应急救援设备，确保在发生意外时能够及时救援。'
#     elif query == "你是谁":
#         response = '我是您的应急安全助手————神机大模型。'
#     elif query == "帮我制定一个开展社区防火知识讲座的计划":
#         response = """好的，开展社区防火知识讲座的计划可以包括以下几个步骤：
#
# 1. 确定讲座的主题和内容，可以包括火灾的成因、预防措施、逃生方法等。
#
# 2. 确定讲座的时间和地点，可以考虑在社区中心、学校、商场等公共场所进行。
#
# 3. 邀请专业的消防人员或相关领域的专家进行讲座，确保内容准确、专业。
#
# 4. 制作宣传海报或传单，提前通知社区居民，吸引更多人参与。
#
# 5. 在讲座中设置互动环节，让居民参与讨论，增强学习效果。
#
# 6. 讲座结束后，可以发放防火知识手册或宣传资料，让居民带回家继续学习。
#
# 7. 定期组织类似的讲座，提高社区居民的防火意识和应急能力。"""
#     else:
#         response = '网络错误!'
    messages = [
        {"role": "user", "content": query['instruction']}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to('cuda')

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1115)
# from flask import Flask, request, jsonify, abort
# # import tensorflow as tf
# from functools import wraps
#
# app = Flask(__name__)
# # model = tf.keras.models.load_model('./model')
#
# API_KEY = '123456'
#
# def require_api_key(func):
#     @wraps(func)
#     def decorated_function(*args, **kwargs):
#         if request.headers.get('x-api-key') != API_KEY:
#             abort(401)
#         return func(*args, **kwargs)
#     return decorated_function
#
# @app.route('/', methods=['POST'])
# # @require_api_key
# def predict():
#     return 'Hello, World!'
#
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, ssl_context=('cert.pem', 'key.pem'))