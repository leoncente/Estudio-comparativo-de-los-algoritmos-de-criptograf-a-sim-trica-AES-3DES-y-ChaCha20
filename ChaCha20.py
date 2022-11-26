import numpy as np

class ChaCha20:
    #Funcion principal para encriptar
    def encrypt(self, txt, key, nonce):
        blocks = np.prod(txt.shape)//(4*16)
        if (np.prod(txt.shape) % (4*16)) != 0:
            blocks = (blocks + 1)

        for c in range(blocks):
            aux = self.generate(key, c+1, nonce)
            txt = self.XOR(txt,aux,c)

        return txt

    #Funcion principal para desencriptar
    def decrypt(self, txt, key, nonce):
        return self.encrypt(txt, key, nonce)

    #Funcion que genera la llave que se aplicara al texto plano
    def generate(self, key, c, nonce):
        block = np.array([0x61707865, 0x3320646e, 0x79622d32, 0x6b206574], dtype=np.uint32)
        block = np.append(block,[key[0], key[1], key[2], key[3], key[4], key[5], key[6], key[7]])
        block = np.append(block,[c, nonce[0], nonce[1], nonce[2]])

        state = np.array(block, copy=True) 

        i = 0
        while i < 10:
            self.quarter_round(block, 0, 4,  8, 12)
            self.quarter_round(block, 1, 5,  9, 13)
            self.quarter_round(block, 2, 6, 10, 14)
            self.quarter_round(block, 3, 7, 11, 15)
            self.quarter_round(block, 0, 5, 10, 15)
            self.quarter_round(block, 1, 6, 11, 12)
            self.quarter_round(block, 2, 7,  8, 13)
            self.quarter_round(block, 3, 4,  9, 14)
            i += 1

        add = np.vectorize(addFun)
        state = add(state, block)

        return state

    #Funcion quarter round
    def quarter_round(self, block, a, b, c, d):
        block[a] = (block[a] + block[b]) % 4294967296
        block[d] = block[d] ^ block[a]
        block[d] = self.rotate(block[d], np.uint32(16))
        block[c] = (block[c] + block[d]) % 4294967296
        block[b] = block[b] ^ block[c]
        block[b] = self.rotate(block[b], np.uint32(12))
        block[a] = (block[a] + block[b]) % 4294967296
        block[d] = block[d] ^ block[a]
        block[d] = self.rotate(block[d], np.uint32(8))
        block[c] = (block[c] + block[d]) % 4294967296
        block[b] = block[b] ^ block[c]
        block[b] = self.rotate(block[b], np.uint32(7))

    #Funcion de reotacion a la izquierda
    def rotate(self, val, can):
        return ((val << can) | (val >> (np.uint32(32) - can))) % 4294967296

    #Funcion que aplica XOR entre el texto plano y un bloque de llave
    def XOR(self, txt, key, c):
        size = np.prod(txt.shape)
        for x in range(64):
            if ((64 * c) + x) < size:
                txt[(64 * c) + x] = txt[(64 * c) + x] ^ self.byte(key, x)
        return txt

    #Funcion que devuelve el byte de la posicion especifica de un arreglos de bytes
    def byte(self, block, pos):
        b = block[pos//4]
        c = (pos % 4) * 8
        b = (b >> c) % 256
        return b

#Funcion que suma modulo 2^32 en notacion Big-Endian
def addFun(a,b):
    return (a + b) % 4294967296