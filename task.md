understand this app, it is a dependency free engine to run qwen 27 gguf 4km . i want to make sure it generates exactly the llama.cpp generation,  i want make make e2e which compare both the genrated text as well (not only the logits) are identical, also the speed is very slow compare to llama.cpp it is now 17 tk/s while llama.cpp 35 tk/s
I am not sure we are using the same way to calc tk/s which llama.cpp uses.
the generated text is diffent i think we supposed to have exact the generated text. 
there are some changes now but i dont know if we should merge them since i feel it affect the generated thinking data 
check also the template is impleted correct as llama.cpp
try make e2e-text-all

