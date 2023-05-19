import torch
from transformers import AutoModel,AutoTokenizer,AutoModelForSeq2SeqLM
from NeuralCom.CoNN.modeling_conn import CoNNModel
from NeuralCom.CoNN import Tokenizer as CoNNTokenizer


class NCModelForCoinFlip(torch.nn.Module):
    def __init__(self, PLM, CoNN, PLMTokenizer, CoNNTokenizer):
        super(NCModelForCoinFlip, self).__init__()
        self.PLM = PLM
        self.CoNN = CoNN
        self.PLMTokenizer = PLMTokenizer
        self.CoNNTokenizer = CoNNTokenizer
        self.d_L = PLM.config.vocab_size
        self.d_C = len(self.CoNNTokenizer.decoder_vocab)
        self.special_word = '->'


    def forward(self, x_L,x_C,output_token):
        x_C = x_C.to(x_L.device)
        H_L = self.PLM(input_ids=x_L,decoder_input_ids=torch.tensor(output_token).to(x_L.device)).logits[:,-1,:]
        H_C = self.CoNN(x_C)[:,-1,:]
        fusion_matrix_L = torch.zeros((self.d_L, self.d_L + self.d_C),device=x_L.device)
        fusion_matrix_L[:, :self.d_L] = torch.eye(self.d_L,device=x_L.device)

        beta = self._get_beta(int(output_token[0][-1]))
        fusion_matrix_C = torch.zeros((self.d_C, self.d_L + self.d_C),device=x_L.device)
        fusion_matrix_C[:, self.d_L:] = beta * torch.eye(self.d_C,device=x_L.device)

        logits = torch.matmul(torch.softmax(H_L,1),fusion_matrix_L) + torch.matmul(H_C,fusion_matrix_C)
        return logits

    def generate(self, x_L):
        self.PLM.eval()
        self.CoNN.eval()
        with torch.no_grad():
            output_token = [torch.tensor([0])]
            for i in range(20):
                x_C = self.tokenizer_encode(output_token)
                logits = self.forward(x_L,x_C,[output_token])
                predicted_tokens = torch.argmax(logits, dim=-1)

                tokens = self.tokenizer_decode(predicted_tokens)
                for token in tokens:
                    output_token.append(token)
                if self.PLMTokenizer.eos_token_id in output_token:
                    break

        return [n.to('cpu').item() for n in output_token]

    def tokenizer_decode(self,predicted_tokens):
        if predicted_tokens >= self.d_L:
            predicted_tokens = predicted_tokens - self.d_L
            return self.PLMTokenizer(self.CoNNTokenizer.decode(predicted_tokens.unsqueeze(0),add_bos=False)[0],return_tensors='pt').input_ids[0,:-1]
        else:
            return [predicted_tokens]

    def tokenizer_encode(self,output_token):
        output = ' '.join(self.PLMTokenizer.batch_decode(output_token))
        output_token = self.CoNNTokenizer(output).unsqueeze(0)
        return output_token

    def _get_beta(self, x):
        # Implement a simple mechanism for determining beta for this example
        if self.special_word in self.PLMTokenizer.convert_ids_to_tokens(x):
            # If the input contains '=', set beta to 1
            return 1
        else:
            # Otherwise, set beta to 0
            return 0





if __name__ == '__main__':
    # Example usage
    PLM = AutoModelForSeq2SeqLM.from_pretrained('WENGSYX/PLM_T5_Base_coin_flip')
    CoNN = CoNNModel.from_pretrained('WENGSYX/CoNN_Parity')
    PLMTokenizer = AutoTokenizer.from_pretrained('WENGSYX/PLM_T5_Base_coin_flip')
    CoNNTokenizer = CoNNTokenizer(CoNN.config.input_encoding_map, CoNN.config.output_encoding_map,CoNN.config.max_position_embeddings)

    neural_comprehension = NCModel(PLM, CoNN, PLMTokenizer, CoNNTokenizer).to('cuda:0')
    input_text = "A coin is heads up. Aaron flips the coin. Julius does not flip the coin. Yixuan Weng flip the coin. Minjun Zhu does not flip the coin. Is the coin still heads up?"
    input_tokens_PLM = PLMTokenizer.encode(input_text, return_tensors='pt')
    generated_output = neural_comprehension.generate(input_tokens_PLM.to('cuda:0'))
    generated_text = PLMTokenizer.decode(generated_output, skip_special_tokens=True)
    print(f"Output: {generated_text}")