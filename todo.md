# TODO List

## High Priority
- [x] Task 1: Correr exp e hyper para outros datasets e frequencias

## Medium Priority
- [x] Task 2: Explorar uso de loss-function-free (já usei uma mas neste caso explorar o do deepseek)

## Low Priority
- [x] Task 3: Explorar outras gates

- [] Task 3: diferentes lags
- [] Task 4: testar com o aux loss


## Notes
- [DeepSeek loss-function-free](https://arxiv.org/pdf/2408.15664v1)
- [DeepSeek code](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py)
- N é preciso correr hyper para todos os hyperparametros, correr apenas para o ultimo horizonte 


- E se usar super supervision no stack. Ao inves de usar apenas a gate, dava scale do output de cada stack e usava-a para calcular o erro com o forecast. 