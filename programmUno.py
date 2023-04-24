# -*- coding: utf-8 -*-
import sys
import nltk
import codecs
from collections import Counter


#funzione per definire corpus, lista di tokens, vocabolario, numero di frasi
def CorpusTokensVocabolario(frasi):
    #creo lista dei tokens
    lista_tokens = []
    #creo numero dei frasi
    tot_frasi = 0
    #Scorro i frasi per frase
    for frase in frasi:
        #divide le frase ai tokens
        tokens = nltk.word_tokenize(frase)
        #creo la lista di tokens e aggiungo elementi in ogni ciclo
        lista_tokens = lista_tokens + tokens
        #creo la lista di frasi e aggiungo elementi in ogni ciclo
        tot_frasi = tot_frasi + 1
    #il mio corpus composta da length di lista dei tokens
    corpus = len(lista_tokens)
    #il mio vocobolario, memorizzare più elementi in una singola variabile
    vocabolario = set(lista_tokens)
    return corpus, lista_tokens, vocabolario, tot_frasi


#funzione per contare frasi e parole
def contaFrasiEToken(frasi, listaToken):
    #dichiaro contatori a 0
    conta_frasi = 0
    conta_parole = 0
    #Scorro i frasi per frase
    for frase in frasi:
        #incrementando conto frasi con ciclo for per avere numeri di frasi
        conta_frasi = conta_frasi + 1  
    for token in listaToken:
        #incrementando conta_parole con ciclo for per avere numeri di token
        conta_parole = conta_parole + len(token) 
    return conta_frasi, conta_parole

#funzione che calcola il numero di hapax sui primi 1000 token di corpus
def hapaxP1000(list_tokens):
    #uso ciclo For per calcolare hapax sui primi 1000 token di corpus
    for porzione in range(1000, 1001):
        #conto hapax di corpus
        tot_hapax = 0
        #lista sara' aggiornato ogni volta con il numero di tokens
        lista_porzioni_incrementali = []
        for i in range(0, porzione):
            lista_porzioni_incrementali.append(list_tokens[i])
        #controllo la frequenza del token e verifico che sia un hapax
        lista_frequenza = Counter(lista_porzioni_incrementali).items()
        for i in lista_frequenza:
            #se un hapax (frequenza = 1) prendi lo
            if i[1] == 1:
                #aggiorna hapax con +1
                tot_hapax = tot_hapax + 1
        #stampa hapax
        print("*** il numero di hapax sui primi 1000 token sono: " +
              str(tot_hapax) + " hapax")
        return tot_hapax

#funzione che calcola vocabolario e TTR
def GranVocabolario(list_tokens):
    #creo la lista vocabolario di incremento 500
    lista_vocabolario500 = []
    #creo la lista di incremento
    lista_incrementale = []
    #creo la lista di vocabolario
    lista_vocabolario = []

    #creo la lista di ciclo pero con incremento di range tra x : x+500  esmp:  2500 : 3000 
    #in stesso tempo anche scorriamo ciclo for con range da 0 a length di list_tokens incremento di 500 (in ogni ciclo aggiorna i numeri perche di x)
    lista_ciclo = [list_tokens[x:x+500] for x in range(0, len(list_tokens),500)]
    for lista in lista_ciclo:
           #incrementando lista di incremento con ciclo for per calcolare lista vocabolario
           lista_incrementale = lista_incrementale + lista
           #con set - (se ci sono elementi occorre piu di una volta setta gli elementi a singolo elemento)
           lista_vocabolario = set(lista_incrementale)
           #conto quanti ci sono nella lista vocabolario
           contatore = len(lista_vocabolario)
           #alla fine metto il numero di vocabolario (+500) alla lista
           lista_vocabolario500.append(contatore)

    return lista_vocabolario500

#funzione che proprio stampare e calcolare TTR di Corpus con incrimento di 500
def stampTTR(list_tokens):
    #creo la lista vocabolario di incremento 500
    lista_vocabolario500 = []
    #creo la lista di incremento
    lista_incrementale = []
    #creo la lista di vocabolario
    lista_vocabolario = []
    #creo il TTR
    TTR = 0.0
     #creo la lista di ciclo pero con incremento di range tra x : x+500  esmp:  2500 : 3000 
    #in stesso tempo anche scorriamo ciclo for con range da 0 a length di list_tokens incremento di 500 (in ogni ciclo aggiorna i numeri perche di x)
    lista_ciclo = [list_tokens[x:x+500] for x in range(0, len(list_tokens),500)]
    for lista in lista_ciclo:
           #incrementando lista di incremento con ciclo for per calcolare lista vocabolario
           lista_incrementale = lista_incrementale + lista
           #con set - (se ci sono elementi occorre piu di una volta setta gli elementi a singolo elemento)
           lista_vocabolario = set(lista_incrementale)
           #conto quanti ci sono nella lista vocabolario
           contatore = len(lista_vocabolario)
           #alla fine metto il numero di vocabolario (+500) alla lista
           lista_vocabolario500.append(contatore)
           #converto numero di corpus al numero intero con float
           corpus = float(len(lista_incrementale))
           #converto numero di vocabolario al numero intero con float
           vocabolario = float(len(lista_vocabolario))
           for lista in range(0, len(lista_vocabolario500),500):
              #stampo TTR con il ciclo di incremento +500 e con il range tra 0 e length di vocabolario500
              TTR = vocabolario/corpus
              print("IL TTR :", TTR)

#funzione che da' il nome del file         
def splitterName(file):
    nome, estensione = (file.name).split(".")  
    return nome

#Note: avevo un problema con collegamento di CorpusTokensVocabolario per quel mottivo ho creato CalcolaToken
def CalcolaToken(frasi):
    tokensTOT=[]
    for frase in frasi:
        tokens=nltk.word_tokenize(frase)                       
        tokensTOT=tokensTOT+tokens                              
    return tokensTOT    

def RappSAAVAPCP(POSTag):
    #Lista dei tag utilizzati nel POS Tagging (Penn Treebank)
    TagSostantivi = ["NN", "NNS", "NNP", "NNPS"]                  
    TagAggettivi = ["JJ", "JJR", "JJS"]
    TagAvverbi = ["RB", "RBR", "RBS"]                  
    TagVerbi = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
    TagArticoli, = ["DT"]                  
    TagPreposizioni = ["IN", "TO"]
    TagCongiunzioni = ["CC"]                  
    TagPronomi = ["PRP", "PRP$"]

    #Numeri di SAAVAPCP
    Numero_Sostantivi = 0
    Numero_Aggettivi = 0
    Numero_Verbi = 0
    Numero_Avverbi = 0
    Numero_Preposizioni = 0
    Numero_Pronomi = 0
    Numero_Articoli = 0
    Numero_Congiunzioni = 0
    
    #creo ciclo For con cui creo if statement per controllare se pos[1] (Tag) e' uno dei S'A'A'V'A'P'C'P , 
    # se accetta condizione aggiorno Numero_XXX con +1
    for pos in POSTag:
        if pos[1] in TagSostantivi:
          Numero_Sostantivi += 1
        if pos[1] in TagVerbi:
          Numero_Verbi += 1
        if pos[1] in TagAggettivi:
          Numero_Aggettivi += 1
        if pos[1] in TagAvverbi:
          Numero_Avverbi += 1
        if pos[1] in TagArticoli:
          Numero_Articoli += 1
        if pos[1] in TagPreposizioni:
          Numero_Preposizioni += 1
        if pos[1] in TagCongiunzioni:
          Numero_Congiunzioni += 1
        if pos[1] in TagPronomi:
          Numero_Pronomi += 1
    #stampo Tutti Percentuali
    print ("Percentuale di Aggettivi sono :",Numero_Aggettivi/100,"%")
    print ("Percentuale di Sostantivi sono :",Numero_Sostantivi/100,"%")
    print ("Percentuale di Verbi sono :",Numero_Verbi/100,"%")
    print ("Percentuale di Avverbi sono :",Numero_Avverbi/100,"%")
    print ("Percentuale di Articoli sono :",Numero_Articoli/100,"%")
    print ("Percentuale di Preposizioni sono :",Numero_Preposizioni/100,"%")
    print ("Percentuale di Congiunzioni sono :",Numero_Congiunzioni/100,"%")
    print ("Percentuale di Pronomi sono :",Numero_Pronomi/100,"%")
    

def main(file1, file2):
    # apre i "file1" e "file2", in sola lettura "r", in codifica "utf-8"
    file1_input = codecs.open(file1, "r","utf-8")
    file2_input = codecs.open(file2, "r","utf-8")  
    
    #nome del file senza estensione
    nome1 = splitterName(file1_input)  
    nome2 = splitterName(file2_input)

    #leggo i file
    riga1 = file1_input.read()
    riga2 = file2_input.read()

    # metodo di lettura del file per la tokenizzazione
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')  
    
    #frasi
    frasi_file1 = sent_tokenizer.tokenize(riga1)  
    frasi_file2 = sent_tokenizer.tokenize(riga2)
    
    #Lunghezza del Corpus, Token e Vocabolario
    lughezza_corpus_file1, listaToken_file1, vocabolario_file1, tot_frasi_file1 = CorpusTokensVocabolario(frasi_file1) 
    lughezza_corpus_file2, listaToken_file2, vocabolario_file2, tot_frasi_file2 = CorpusTokensVocabolario(frasi_file2)

    #numero di token totale di corpus
    tot_token_file1 = len(listaToken_file1)
    tot_token_file2 = len(listaToken_file2)
    
    #richiamo funzione contaFrasiEToken() per conteggio frasi e caratteri
    numero_tot_frasi_file1, tot_caratteri_file1 = contaFrasiEToken(frasi_file1, listaToken_file1)  
    numero_tot_frasi_file2, tot_caratteri_file2 = contaFrasiEToken(frasi_file2, listaToken_file2)

    #lunghezza media delle frasi in termini di token
    lunghezza_media_frasi_file1 = tot_token_file1 / numero_tot_frasi_file1  
    lunghezza_media_frasi_file2 = tot_token_file2 / numero_tot_frasi_file2
    
    #lunghezza media delle parole in termini di caratteri
    lunghezza_media_parole_file1 = tot_caratteri_file1 / tot_token_file1  
    lunghezza_media_parole_file2 = tot_caratteri_file2 / tot_token_file2

    #vocabolario incremento +500
    vocabolario500_file1 = GranVocabolario(listaToken_file1)
    vocabolario500_file2 = GranVocabolario(listaToken_file2)
    
    #Note: avevo un problema con collegamento di CorpusTokensVocabolario per quel mottivo ho creato CalcolaToken,
    #Calcolo i token su tutto il corpus
    Corpus1 = CalcolaToken(frasi_file1) 
    Corpus2 = CalcolaToken(frasi_file2)

    #tokenPOS uso per calcolare Percentuali di S'A'A'V'A'P'C'P
    tokensPOS1 = nltk.pos_tag(Corpus1) 
    tokensPOS2 = nltk.pos_tag(Corpus2) 


    print("Iskandar Huseynzade ( 618017 )")
    print("________________________________")
    # RISULTATI **************************************************
    print("\nIl confronto tra due corpus ", nome1, ".txt ,", nome2, ".txt\n")
    print("- CONFRONTI BASILARI -")
    print("\n", nome1, "\t\t\t", nome2, "\n")
    #NUMERI DI FRASI E NUMERI DI TOKEN
    print("Frasi file1 -->", numero_tot_frasi_file1, "\t\tFrasi file2 -->", numero_tot_frasi_file2)  
    print("Token file1 -->", tot_token_file1, "\t\tToken file2", tot_token_file2, "\n") 
    print("Media frasi file1 -->", lunghezza_media_frasi_file1, "\tMedia frasi file2 -->", lunghezza_media_frasi_file2)  
    print("Media token file1 -->", lunghezza_media_parole_file1,"\tMedia token file2 -->", lunghezza_media_parole_file2,"\n")  
    if numero_tot_frasi_file1 > numero_tot_frasi_file2:
        print("*** Il numero totale delle frasi scritte in", nome1, "è maggiore di quelle scritte in", nome2)
    elif numero_tot_frasi_file1 == numero_tot_frasi_file2:
        print("*** Il numero totale delle frasi in", nome1, "e di quelle in", nome2, ", è lo stesso")
    else:
        print("*** Il numero totale delle frasi in", nome1, "è maggiore a quelle scritte in", nome2)
    if tot_token_file1 > tot_token_file2:
        print("*** In corpus", nome1,"è stata usata piu numero di token rispetto al ", nome2)
    elif tot_token_file1 == tot_token_file2:
        print("*** I numeri di token tra due corpus è uguale")
    else:
        print("*** In corpus", nome2,"è stata usata piu numero di token rispetto al ", nome1)
    # confronto lunghezza media delle frasi
    if lunghezza_media_frasi_file1 > lunghezza_media_frasi_file2:
        print( "*** Il corpus di", nome1,"ha una lunghezza media delle frasi, in termini di token, più rispetto al corpus di",nome2)
    elif lunghezza_media_frasi_file1 == lunghezza_media_frasi_file2:
        print("*** La lunghezza media delle frasi in termini di token di due corpus sono uguale")
    else:
        print( "*** Il corpus di", nome2,"ha una lunghezza media delle frasi, in termini di token, più rispetto al corpus di",nome1)
    # confronto lunghezza media delle token
    if lunghezza_media_parole_file1 > lunghezza_media_parole_file2:
        print( "*** Il corpus di", nome1, "ha una lunghezza media delle token, in termini di token, più rispetto al corpus di",nome2, "\n")
    elif lunghezza_media_parole_file1 == lunghezza_media_parole_file2:
        print( "*** Lunghezza media di token in termini di caratteri di entrambi due corpus sono uguale","\n")
    else:
        print("*** Il corpus di", nome2,"ha unalunghezza media delle token, in termini di token, più rispetto al corpus di",nome1, "\n")


    # Numero di hapax sui primi 1000 token
    print("NUMERO DI HAPAX sui primi 1000 token")
    print("Il corpus di" + nome1)
    hapaxP1000(listaToken_file1)
    print("Il corpus di" + nome2)
    hapaxP1000(listaToken_file2)
    print("\n")
    
    # La Grandezza del Vocabolario Incremento di +500
    print("La Grandezza del Vocabolario +500")
    print(nome1, "\t", nome2)
    for f1, f2 in zip(vocabolario500_file1, vocabolario500_file2):
      print("- %-10s" %(f1), "- %-10s" % (f2))
    # Il TTR con incremento di +500
    print("\n", "Il Type Token Ratio per file", nome1, "con incremento +500 sono :")
    stampTTR(listaToken_file1)
    print("\n", "Il Type Token Ratio per file", nome1, "con incremento +500 sono :")
    stampTTR(listaToken_file2)
    
    # Percentuali di Aggettivi, Sostantivi,Verbi, Avverbi, Articoli, Preposizioni, Congiunzioni, Pronomi.
    print("\n","Il corpus di", nome1, "ha sequenti percentuali")
    print()
    RappSAAVAPCP(tokensPOS1)
    print("\n","Il corpus di", nome2, "ha sequenti percentuali")
    print()
    RappSAAVAPCP(tokensPOS2)

main(sys.argv[1], sys.argv[2])
