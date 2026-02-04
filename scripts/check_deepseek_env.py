import os
print('has_key', bool(os.environ.get('DEEPSEEK_API_KEY')))
print('has_base', bool(os.environ.get('DEEPSEEK_BASE_URL')))
