from typing import Union, Dict, TypeVar
import logging, os
from app.args import Args
from app.client.base_manager import BaseAIManager
from app.client.llm import LLMClient_LlamaCppPython
from app.models.llm.llm_api import CompletionReturn
from app.models.llm.client import CompletionOptions, CompletionOptions_LlamaCppPython

ClientDict = Dict[str, LLMClient_LlamaCppPython]

EXTENSIONS = ['.gguf', '.ggml', '.safetensor']
logger = logging.getLogger(__name__)

class LLMManager(BaseAIManager):
	clients: ClientDict = {
		'llamacpp': LLMClient_LlamaCppPython.instance,
	}
	loader: Union[LLMClient_LlamaCppPython, None] = None

	def __init__(self):
		self.clients = {
			'llamacpp': LLMClient_LlamaCppPython.instance,
		}
		self.default_model = Args['llm_model']
		self.models_dir = Args['llm_models_dir']

	def get_loader_model(self):
		if self.loader == LLMClient_LlamaCppPython.instance:
			return CompletionOptions_LlamaCppPython

	def list_local_models(self) -> list[str]:
		models_dir = self.get_models_dir()
		if not models_dir or not os.path.isdir(models_dir):
			return []
		model_names = []
		for filename in os.listdir(models_dir):
			path = os.path.join(models_dir, filename)
			if os.path.isfile(path) and os.path.splitext(filename)[
				1] in EXTENSIONS:
				model_names.append(filename)
			if os.path.isdir(path):
				f = filename.lower()
				if 'awq' in f or 'gptq' in f or 'exl2' in f:
					model_names.append(filename)
					continue
				for subfilename in os.listdir(
					os.path.join(models_dir, filename)
				):
					path = os.path.join(models_dir, filename, subfilename)
					if os.path.isfile(path) and os.path.splitext(
						subfilename
					)[1] in EXTENSIONS:
						model_names.append(
							os.path.join(filename, subfilename)
						)
		return model_names

	def list_models(self) -> list[str]:
		# TODO list models in dir
		models = []
		return models

	def pick_client(self, model_name: str):
		return 'llamacpp'

	def get_models_dir(self):
		return Args['llm_models_dir']

	def get_default_model(self):
		return Args['llm_model']

	def load_model(self, model_name: Union[str, None]):
		if model_name is None or model_name == '':
			return super().load_model(model_name)
		if 'openai:' in model_name:
			self.model_name = model_name
			self.loader_name = 'openai'
			return

		return super().load_model(model_name)

	def generate(self, gen_options: CompletionOptions):
		if not self.loader:
			self.load_model(None)
			if not self.loader:
				raise Exception('Model not loaded.')
		options = self.loader.convert_options(gen_options)
		loader_model = self.get_loader_model()
		assert loader_model is not None
		assert isinstance(options, loader_model)
		return self.loader.generate(options)  # type: ignore

	def chat(self, options):
		model = options['model']
		if model is None or model == '':
			model = self.model_name or Args['llm_model']
			options['model'] = model

		if not self.loader:
			self.load_model(model or None)
		if not self.loader:
			raise Exception('Model not loaded.')
		# new_options = self.loader.convert_options(options)
		loader_model = self.get_loader_model()
		assert loader_model is not None
		# assert isinstance(new_options, loader_model)
		result = self.loader.chat(options)
		return result

	def complete(self, gen_options: CompletionOptions):
		model = gen_options.model
		if model is None or model == '':
			model = self.model_name or Args['llm_model']
			gen_options.model = model

		if not self.loader:
			self.load_model(model or None)
		if not self.loader:
			raise Exception('Model not loaded.')
		options = self.loader.convert_options(gen_options)
		loader_model = self.get_loader_model()
		assert loader_model is not None
		assert isinstance(options, loader_model)
		result = self.loader.complete(options)  # type: ignore

		try:
			validated_result = CompletionReturn.model_validate(result)
		except Exception as e:
			logger.error(
				"Validation failed for model %s with options %s. Error: %s",
				model, gen_options, str(e)
			)
			return None

		return validated_result
