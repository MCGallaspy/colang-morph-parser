import torch

def encode_input(word, input_dict):
    idxs = []
    for c in word:
        try:
            idxs.append(input_dict[c])
        except KeyError:
            idxs.append(input_dict["UNK"])
    result = []
    for idx in idxs:
        t = torch.zeros(len(input_dict))
        t[idx] = 1
        result.append(t)
    result = torch.stack(result)
    if result.dim() == 1:
        result = result.unsqueeze(0)
    return result

def decode_output(seq, output_alphabet_reversal):
    result = []
    for subseq in seq:
        idx = int(torch.argmax(subseq))
        result.append(output_alphabet_reversal[idx])
    return result

def get_output(model, seq, output_alphabet):
    result = [torch.Tensor([-99999] * len(output_alphabet))]
    result[0][output_alphabet["GLOSS_START"]] = 0
    MAX_LEN = 5
    model.eval()
    with torch.no_grad():
        for _ in range(MAX_LEN):
            mask = torch.nn.Transformer.generate_square_subsequent_mask(len(result))
            next = model(seq, torch.stack(result), tgt_mask=mask, tgt_is_causal=True)
            argmax = torch.argmax(next[-1])
            result.append(next[-1])
            if argmax == output_alphabet["GLOSS_END"]:
                break
    
    return torch.stack(result)
    st.write([output_reversal[int(torch.argmax(y))] for y in result])

def get_entropy(log_probs):
    probs = torch.exp(log_probs)
    return torch.sum(probs * log_probs * -1)
