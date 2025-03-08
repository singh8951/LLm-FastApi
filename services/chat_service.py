class ChatService:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate_response(self, prompt, max_length=50):
        # Tokenize input prompt
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        
        # Generate text
        outputs = self.model.generate(inputs, max_length=max_length, num_return_sequences=1)
        
        # Decode and return the generated text
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"response": response}