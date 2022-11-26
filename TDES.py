import numpy as np

#Tabla para la permutacion IP
IP = [57, 49, 41, 33, 25, 17, 9,  1,
      59, 51, 43, 35, 27, 19, 11, 3,
      61, 53, 45, 37, 29, 21, 13, 5,
      63, 55, 47, 39, 31, 23, 15, 7,
      56, 48, 40, 32, 24, 16, 8,  0,
      58, 50, 42, 34, 26, 18, 10, 2,
      60, 52, 44, 36, 28, 20, 12, 4,
      62, 54, 46, 38, 30, 22, 14, 6]

#Tabla para la permutacion IP-1
invIP = [39,  7, 47, 15, 55, 23, 63, 31,
         38,  6, 46, 14, 54, 22, 62, 30,
         37,  5, 45, 13, 53, 21, 61, 29,
         36,  4, 44, 12, 52, 20, 60, 28,
         35,  3, 43, 11, 51, 19, 59, 27,
         34,  2, 42, 10, 50, 18, 58, 26,
         33,  1, 41,  9, 49, 17, 57, 25,
         32,  0, 40,  8, 48, 16, 56, 24]

#Tabla para la funcion de expansion E
E = [31,  0,  1,  2,  3,  4,
      3,  4,  5,  6,  7,  8,
      7,  8,  9, 10, 11, 12,
     11, 12, 13, 14, 15, 16,
     15, 16, 17, 18, 19, 20,
     19, 20, 21, 22, 23, 24,
     23, 24, 25, 26, 27, 28,
     27, 28, 29, 30, 31,  0]

#Tabla para la permutacion PC-1
PC1 = [56, 48, 40, 32, 24, 16,  8,
        0, 57, 49, 41, 33, 25, 17,
        9,  1, 58, 50, 42, 34, 26,
       18, 10,  2, 59, 51, 43, 35,
       62, 54, 46, 38, 30, 22, 14,
        6, 61, 53, 45, 37, 29, 21,
       13,  5, 60, 52, 44, 36, 28,
       20, 12,  4, 27, 19, 11,  3]

#Tabla para la permutacion PC-2
PC2 = [13, 16, 10, 23,  0,  4,
        2, 27, 14,  5, 20,  9,
       22, 18, 11,  3, 25,  7,
       15,  6, 26, 19, 12,  1,
       40, 51, 30, 36, 46, 54,
       29, 39, 50, 44, 32, 47,
       43, 48, 38, 55, 33, 52,
       45, 41, 49, 35, 28, 31]

#S1
S = [[14,  4, 13,  1,  2, 15, 11,  8,  3, 10,  6, 12,  5,  9,  0,  7,
       0, 15,  7,  4, 14,  2, 13,  1, 10,  6, 12, 11,  9,  5,  3,  8,
       4,  1, 14,  8, 13,  6,  2, 11, 15, 12,  9,  7,  3, 10,  5,  0,
      15, 12,  8,  2,  4,  9,  1,  7,  5, 11,  3, 14, 10,  0,  6, 13],
#S2
     [15,  1,  8, 14,  6, 11,  3,  4,  9,  7,  2, 13, 12,  0,  5, 10,
       3, 13,  4,  7, 15,  2,  8, 14, 12,  0,  1, 10,  6,  9, 11,  5,
       0, 14,  7, 11, 10,  4, 13,  1,  5,  8, 12,  6,  9,  3,  2, 15,
      13,  8, 10,  1,  3, 15,  4,  2, 11,  6,  7, 12,  0,  5, 14,  9],
#S3
     [10,  0,  9, 14,  6,  3, 15,  5,  1, 13, 12,  7, 11,  4,  2,  8,
      13,  7,  0,  9,  3,  4,  6, 10,  2,  8,  5, 14, 12, 11, 15,  1,
      13,  6,  4,  9,  8, 15,  3,  0, 11,  1,  2, 12,  5, 10, 14,  7,
       1, 10, 13,  0,  6,  9,  8,  7,  4, 15, 14,  3, 11,  5,  2, 12],
#S4
     [ 7, 13, 14,  3,  0,  6,  9, 10,  1,  2,  8,  5, 11, 12,  4, 15,
      13,  8, 11,  5,  6, 15,  0,  3,  4,  7,  2, 12,  1, 10, 14,  9,
      10,  6,  9,  0, 12, 11,  7, 13, 15,  1,  3, 14,  5,  2,  8,  4,
       3, 15,  0,  6, 10,  1, 13,  8,  9,  4,  5, 11, 12,  7,  2, 14],
#S5
     [ 2, 12,  4,  1,  7, 10, 11,  6,  8,  5,  3, 15, 13,  0, 14,  9,
      14, 11,  2, 12,  4,  7, 13,  1,  5,  0, 15, 10,  3,  9,  8,  6,
       4,  2,  1, 11, 10, 13,  7,  8, 15,  9, 12,  5,  6,  3,  0, 14,
      11,  8, 12,  7,  1, 14,  2, 13,  6, 15,  0,  9, 10,  4,  5,  3],
#S6
     [12,  1, 10, 15,  9,  2,  6,  8,  0, 13,  3,  4, 14,  7,  5, 11,
      10, 15,  4,  2,  7, 12,  9,  5,  6,  1, 13, 14,  0, 11,  3,  8,
       9, 14, 15,  5,  2,  8, 12,  3,  7,  0,  4, 10,  1, 13, 11,  6,
       4,  3,  2, 12,  9,  5, 15, 10, 11, 14,  1,  7,  6,  0,  8, 13],
#S7
     [ 4, 11,  2, 14, 15,  0,  8, 13,  3, 12,  9,  7,  5, 10,  6,  1,
      13,  0, 11,  7,  4,  9,  1, 10, 14,  3,  5, 12,  2, 15,  8,  6,
       1,  4, 11, 13, 12,  3,  7, 14, 10, 15,  6,  8,  0,  5,  9,  2,
       6, 11, 13,  8,  1,  4, 10,  7,  9,  5,  0, 15, 14,  2,  3, 12],
#S8
     [13,  2,  8,  4,  6, 15, 11,  1, 10,  9,  3, 14,  5,  0, 12,  7,
       1, 15, 13,  8, 10,  3,  7,  4, 12,  5,  6, 11,  0, 14,  9,  2,
       7, 11,  4,  1,  9, 12, 14,  2,  0,  6, 10, 13, 15,  3,  5,  8,
       2,  1, 14,  7,  4, 10,  8, 13, 15, 12,  9,  0,  3,  5,  6, 11]]

#Tabla para la funcion P
P = [15,  6, 19, 20,
     28, 11, 27, 16,
      0, 14, 22, 25,
      4, 17, 30,  9,
      1,  7, 23, 13,
     31, 26,  2,  8,
     18, 12, 29,  5,
     21, 10,  3, 24]
class TDES:
    #Funcion principal para encriptar
    def encrypt(self, txt, key1, key2, key3):
        txt = self.encryptDES(txt, key1)
        txt = self.decryptDES(txt, key2)
        txt = self.encryptDES(txt, key3)

        return txt

    #Funcion principal para desencriptar
    def decrypt(self, txt, key1, key2, key3):
        txt = self.decryptDES(txt, key3)
        txt = self.encryptDES(txt, key2)
        txt = self.decryptDES(txt, key1)

        return txt

    #Funcion de encriptacion DES
    def encryptDES(self, txt, key):
        tam = np.prod(txt.shape)
        if tam % 64 != 0:
            for x in range(64 - (tam % 64)):
                txt = np.append(txt, 0)
        tam = np.prod(txt.shape)
        res = np.empty(tam, dtype=np.uint8)
        k = self.KS(key)
        for x in range(tam//64):
            block = txt[x*64:(x*64)+64]
            self.funIP(block)
            L = block[0:32]
            R = block[32:64]
            for y in range(16):
                aux = R
                R = self.XOR(L, self.F(R,k[y]))
                L = aux
            block = np.concatenate((R, L))
            self.funInvIP(block)
            res[x*64:(x*64)+64] = block
        return res

    #Funcion de desencriptacion DES
    def decryptDES(self, txt, key):
        tam = np.prod(txt.shape)
        res = np.empty(tam, dtype=np.uint8)
        k = self.KS(key)
        for x in range(tam//64):
            block = txt[x*64:(x*64)+64]
            self.funIP(block)
            R = block[0:32]
            L = block[32:64]
            for y in range(16):
                aux = L
                L = self.XOR(R, self.F(L,k[15-y]))
                R = aux
            block = np.concatenate((L, R))
            self.funInvIP(block)
            res[x*64:(x*64)+64] = block
        t = np.prod(res.shape)
        c = 0
        aux1 = 0
        f = True
        aux2 = t - 1
        while f:
            if(res[aux2] == 0):
                aux1 += 1
                if aux1 == 8:
                    aux1 = 0
                    c += 1
            else:
                f = False
            aux2 -= 1
        return res[0:t - (c*8)]

    #Funcion IP
    def funIP(self, block):
        block[0:] = np.asarray([block[x] for x in IP])

    #Funcion IP-1
    def funInvIP(self, block):
        block[0:] = np.asarray([block[x] for x in invIP])

    #Funcion KS
    def KS(self, key):
        k = []
        key = self.funPC1(key)
        C = key[0:28]
        D = key[28:56]
        for x in range(16):
            if(x == 0 or x == 1 or x == 8 or x == 15 ):
                C = self.RI(C)
                D = self.RI(D)
            else:
                C = self.RI(C)
                D = self.RI(D)
                C = self.RI(C)
                D = self.RI(D)
            aux = C + D
            aux = self.funPC2(aux)
            k.append(aux)
        return k

    #Funcion PC1
    def funPC1(self, key):
        return [key[x] for x in PC1]

    #Funcion PC2
    def funPC2(self, key):
        return [key[x] for x in PC2]

    #Funcion de rotacion a la izquierda
    def RI(self, block):
        res = block[1:28]
        res.append(block[0])
        return res

    #Funcion que hace XOR sobre dos listas
    def XOR(self, l1, l2):
        c = 0
        res = []
        for x in l2:
            res.append(l1[c] ^ x)
            c += 1
        return np.asarray(res)

    #Funcion F
    def F(self, R, k):
        R = self.funE(R)
        R = self.XOR(R,k)
        res = self.funS(R[0:6],0)+ self.funS(R[6:12],1)+ self.funS(R[12:18],2)+ self.funS(R[18:24],3)+ self.funS(R[24:30],4)+ self.funS(R[30:36],5)+ self.funS(R[36:42],6)+ self.funS(R[42:48],7)
        res = self.funP(res)
        return res

    #Funcion E
    def funE(self, block):
        res = [block[x] for x in E]
        return np.asarray(res)

    #Funcion P
    def funP(self, block):
        res = [block[x] for x in P]
        return np.asarray(res)

    #Funcion que mapea las tablas S
    def funS(self, block, n):
        row = block[0]*2 + block[5]
        col = block[1]*8 + block[2]*4 + block[3]*2 + block[4]
        num = (row * 16) + col
        aux = S[n][num]
        res = [0,0,0,0]
        res[3] = (aux % 2)
        aux = aux//2
        res[2] = (aux % 2)
        aux = aux//2
        res[1] = (aux % 2)
        aux = aux//2
        res[0] = (aux % 2)
        return res