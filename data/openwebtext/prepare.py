from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset, load_dataset_builder, get_dataset_split_names # dataset de huggingface

# numero de workers en .map() call
# un buen numero para usar es ~order numero de cpu cores // 2
num_proc = 8

# En primer lugar vamos a inspeccionar la descripcion, features, citation y el homepage del dataset Openwebtext
ds_builder = load_dataset_builder("openwebtext")

print("Descripcion de OWT: \n", ds_builder.info.description, "\n")
print("Features de OWT: \n", ds_builder.info.features, "\n")
print("Cita de OWT: \n", ds_builder.info.citation, "\n")
print("Sitio Web de OWT: \n", ds_builder.info.homepage, "\n")

# Obtenemos los subconjuntos del OWT
print("--Subconjuntos--", get_dataset_split_names("openwebtext"))

# tomamos 54GB  en huggingface .cache dir, cerca de 8M de documentos (8,013,769) .cache/huggingface/datasets/openwebtext/plain_text/1.0.0/
dataset = load_dataset("openwebtext")
print("---Todos los subconjuntos y features---", dataset)


#openwebtext o owt por defecto sólo contiene el 'train' split, asi que creamos un test split
split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
# cuando aplicamos un train_test_split dividimos nuestro dataset en dos partes un llamado train y otro test
# mirad el resultado
# DatasetDict({
#     train: Dataset({
#         features: ['text'],
#         num_rows: 8009762
#     })
#     test: Dataset({
#         features: ['text'],
#         num_rows: 4007
#     })
# })

# pero ahora usaremos la funcion pop para eliminar el dataset test y lo agregamos como dataset de validacio{on al cual lo llamamos val
split_dataset['val'] = split_dataset.pop('test') # renombramos el test split a val

# mostramos en consola ambos dataset train y val
print(split_dataset)

# El resultado de este print a split_dataset será: 
# DatasetDict({
#     train: Dataset({
#         features: ['text'],
#         num_rows: 8009762
#     })
#     val: Dataset({
#         features: ['text'],
#         num_rows: 4007
#     })
# })


# vamos a tokenizar el dataset. primero definimos the funcion de encoding (gpt2 bpe)
enc = tiktoken.get_encoding('gpt2')

def process(example):
    ids = enc.encode_ordinary(example['text']) # enconde_ordinary ignora cualquier token especial
    ids.append(enc.eot_token) # Al final del texto agregamos el token '50256 o <|endoftext|> , para gpt2 bpe
    # nota: EOT es un acrónimo que se End Of Text o End Of Transmission 
    out = {'ids': ids, 'len': len(ids)} # creamos un diccionario llamado out con dos elementos, uno id y el otro len
    return out

# Para ilustrar lo que hace la funcion de arriba process

# text= "Hola, me llamo Alexander y tú como te llamas?"

# ids= enc.encode_ordinary(text)
# print(ids)
# Estos son los tokens que devuelve encode_ordinary: [39, 5708, 11, 502, 32660, 18811, 10009, 331, 256, 21356, 401, 78, 573, 32660, 17485, 30]

#ids.append(enc.eot_token)
#print(ids)
# Vemos que al final se agrego el token 50256 : [39, 5708, 11, 502, 32660, 18811, 10009, 331, 256, 21356, 401, 78, 573, 32660, 17485, 30, 50256]

#print(enc.decode(ids))
# El token 50026 significa <|endoftext|> : Hola, me llamo Alexander y tú como te llamas?<|endoftext|>

#out = {'ids': ids, 'len': len(ids)}
#print(out)
#'ids': [39, 5708, 11, 502, 32660, 18811, 10009, 331, 256, 21356, 401, 78, 573, 32660, 17485, 30, 50256], 'len': 17}


# Aqui aplicamos la funcion process al dataset split_dataset creado lineas arriba
tokenized = split_dataset.map(
    process, # Funcion de tokenizado
    remove_columns=['text'], # Luego de aplicar la función al dataset elimnamos la columna text
    desc="tokenizing the splits", # Descripción que se mostrará en la barra de progreso
    num_proc=num_proc # Numero de procesos para generar un dataset local
)

#tokenizing the splits #0: 100%|█████████████████████████████████████████████████████████████████████████████| 501/501 [00:01<00:00, 392.79ex/s]
#tokenizing the splits #6: 100%|█████████████████████████████████████████████████████████████████████████████| 501/501 [00:01<00:00, 386.31ex/s]
#tokenizing the splits #5: 100%|█████████████████████████████████████████████████████████████████████████████| 501/501 [00:01<00:00, 377.24ex/s]
#tokenizing the splits #7: 100%|█████████████████████████████████████████████████████████████████████████████| 500/500 [00:01<00:00, 372.65ex/s]
#tokenizing the splits #3: 100%|█████████████████████████████████████████████████████████████████████████████| 501/501 [00:01<00:00, 376.59ex/s]
#tokenizing the splits #4: 100%|█████████████████████████████████████████████████████████████████████████████| 501/501 [00:01<00:00, 364.67ex/s]
#tokenizing the splits #2: 100%|█████████████████████████████████████████████████████████████████████████████| 501/501 [00:01<00:00, 362.56ex/s]
#tokenizing the splits #1: 100%|█████████████████████████████████████████████████████████████████████████████| 501/501 [00:01<00:00, 360.43ex/s]

print(tokenized)

#    train: Dataset({
#        features: ['ids', 'len'],
#        num_rows: 8009762
#    })
#    val: Dataset({
#        features: ['ids', 'len'],
#        num_rows: 4007
#    })
#})

# concatenamos todos los ids en cada dataset en un solo archivo que nosotros podemos usar para el training

for split, dset in tokenized.items():
    print(split)
    print(dset)
    arr_len = np.sum(dset['len'])
    filename = f'{split}.bin' # Acá usamos formatspec de python para crear un archivo train.bin y val.bin
    dtype = np.uint16 # Definimos este tipo para numeros enteros sin signo en el rango de 0 a 65535 (2^16 - 1) y como el valor maximo del token eot es 50256  2^16 - 1 
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,)) # en la variable arr es del tipo memoria mapeada, esto es util porque se trabaja con archivos grandes 

    print(f"writing {filename}...") # esto indica el nombre del archivo en el que se esta escribiendo la matriz por ejemplo train.bin o val.bin
    idx = 0 # establecemos el valor 0 para iniciar
    for example in tqdm(dset): # con tqdm tendremos una barra de progreso segun vayamos iterando sobre cada elemento de dset
        arr[idx : idx + example['len']] = example['ids'] # esto es slicing para asignar los valores de example['ids'] a una sección de la matriz arr
        idx += example['len'] # sumamos el valor actual de example['len']
    arr.flush() # por ultimo usamos la función flush para vaciar el buffer, es decir escribiremos fisicamente todos los datos en el disco duro.

# train.bin pesa ~17GB, val.bin ~8.5MB
# train has ~9B tokens (9,035,582,198)
# val tiene ~4M tokens (4,434,897)

# Luego leeremos los archivos .bin con numpy de la siguiente manera:
# m = np.memmap('train.bin', dtype=np.uint16, mode='r')