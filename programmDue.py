
from re import S
import sys
import codecs  # classe che contiene il metodo open()
import nltk  # per metodo tokenize
import collections  # per collections.Counter()
import math  # per log() di LMI
from nltk import bigrams

ListaPunteggiatura = [".", ",", ":", ";", "!", "?", "/", "..", "-", "(", ")", "'","’","“","”", "_","-","SYM",]


#funzione per definire lista di tokens, lista di POS
def tokenizePOS(frasi):
    #creo lista dei tokens
    lista_tokens = []
    #lista dei POS 
    lista_POS = []
    #Scorro i frasi per frase
    for frase in frasi:
        #divido in token la frase
        tokens = nltk.word_tokenize(frase)
        #il POS Tag dei token
        tokensPOS = nltk.pos_tag(tokens)
        #Calcolo tutti token di corpus
        lista_tokens+=tokens
        #calcolo POS di corpus
        lista_POS+=tokensPOS

    return lista_tokens, lista_POS

#funzione calcola lista token senza la punteggiatura
def tokNoPunct(tokensList):
    tokListNoPunct = []
    punct =  [".", ",", ":", ";", "!", "?", "/", "..", "-", "(", ")", "'","’","“","”", "_","-","SYM",]
    #uso ciclo For per controllare token in ogni ciclo
    for token in tokensList:
        #se non un punteggiatura aggiungo la nuova lista 
        if token not in punct:
            tokListNoPunct.append(token)

    return tokListNoPunct


#funzione che calcola to 20 elementi più frequenti ordine descrescente rispetto alla frequenza
def top20FreqDecrs(elemList):
    # calcolo frequenza di ogni elemento nella lista
    freq = collections.Counter(elemList)
    # creo variabile che ha primi 20 elem in ordine decrescente
    most20Freq = freq.most_common(20)

    return most20Freq


# funzione che calcola liste di avverbi e aggettivi più frequenti
def mostFreqAvvAgg(tokensPOS):
    # le liste vuote
    mostFreqAvv = []
    mostFreqAgg = []
    # liste POS tags
    # link: https://elearning.humnet.unipi.it/pluginfile.php/319388/mod_page/content/41/Penn%20Treebank%20tagset.pdf
    tagAvverbi = ["RB", "RBR", "RBS"]
    tagAggettivi = ["JJ", "JJR", "JJS"]
    # per ogni elemento, formato dai token-tag, guardo se tag è uguale a Avverbo o Aggettivo
    # elemento[0] mostrare token (parola)
    # elemento[1] Tag di token 
    for elemento in tokensPOS:
        if elemento[1] in tagAvverbi and elemento[0] not in ListaPunteggiatura:
            # se un tag di Avverbo aggiungo elemento alla lista dagli Avverbi
            mostFreqAvv.append(elemento[0])
        if elemento[1] in tagAggettivi and elemento[0] not in ListaPunteggiatura:
            # se un tag di Avverbo aggiungo elemento alla lista dagli Aggettivi
            mostFreqAgg.append(elemento[0])

    return mostFreqAvv, mostFreqAgg


# funzione che mette ordine Aggettivi e Avverbi
def soloAggAvv(tokensPOS):
    # creo lista bigrammi di POS
    listaBigrammi = nltk.bigrams(tokensPOS)
    # link: https://elearning.humnet.unipi.it/pluginfile.php/319388/mod_page/content/41/Penn%20Treebank%20tagset.pdf
    listaTag = ["JJ", "JJR", "JJS","RB", "RBR", "RBS"]
    # creo nuova lista bigrammi i cui tag non sono tra quelli da evitare
    listaBigrNew = []
    for ((tok1, tag1), (tok2, tag2)) in listaBigrammi:
        if tag1 not in listaTag:
            if tag2 not in listaTag:
                bigramma = (tok1, tok2)
                listaBigrNew.append(bigramma)

    return listaBigrNew


# funzione che crea lista di tag POS per 10 POS più frequenti
def listaTagPOS(tokensPOS):
    #lista Tag
    listaTag = []
    for (tok, tag) in tokensPOS:
        # se POS non sono Punteggiature
        if not tok[0] in ListaPunteggiatura:
            #aggiungo la lista 
            listaTag.append(tag)
    return listaTag


# funzione che calcola 10 elementi più frequenti e con frequenza decrescente
def elem10PiuFreqDecresc(elemList):
    # calcolo frequenza di ogni elemento nella lista
    freq = collections.Counter(elemList)
    #assegno primi 10 elem in ordine decrescente
    top10Freq = freq.most_common(10)

    return top10Freq


# funzione che costruisce lista bigrammi (di tag) POS
def POSbigrammi(tokensPOS):
    # creo lista bigrammi di POS
    listaBigrammi = nltk.bigrams(tokensPOS)
    # creo nuova lista bigrammi con i tag POS
    listaBigrNew = []
    for ((tok1, tag1), (tok2, tag2)) in listaBigrammi:
        # con ciclo For controllo ogni volta se tags non sono punteggiatura
        if tag1 not in ListaPunteggiatura:
            if tag2 not in ListaPunteggiatura:
                #aggiungo bigrammi alla lista
                bigramma = (tag1, tag2)
                listaBigrNew.append(bigramma)

    return listaBigrNew


#funzione che costruisco lista trigrammi (di tag) POS
def POStrigrammi(tokensPOS):
    # creo lista trigrammi di POS
    listaTrigrammi = nltk.trigrams(tokensPOS)
    # creo nuova lista trigrammi con i tag POS
    listaTrigrNew = []
    for ((tok1, tag1), (tok2, tag2), (tok3, tag3)) in listaTrigrammi:
        # con ciclo For controllo ogni volta se tags non sono punteggiatura
        if tag1 not in ListaPunteggiatura:
            if tag2 not in ListaPunteggiatura:
                if tag3 not in ListaPunteggiatura:
                    #aggiungo trigrammi alla lista
                    trigramma = (tag1, tag2, tag3)
                    listaTrigrNew.append(trigramma)

    return listaTrigrNew


# funzione che costruisce lista composta da aggettivi e sostantivi con frequenza maggiore di 3
def aggSost(tokensPOS):
    # lista Aggettivi e Sostantivi
    listaNomiAgg = []
    # liste POS tags
    tagNomi = ["NN", "NNS", "NNP", "NNPS"]
    tagAggettivi = ["JJ", "JJR", "JJS"]
    # per ogni elemento, guardo se il secondo(il tag) è uguale a uno di quelli delle liste di Tag Nomi o Tag Aggettivi
    for elem in tokensPOS:
        # calcolo frequenza ogni token, condizione frequenza maggiore di 3
        freq = tokensPOS.count(elem)
        if freq > 3:
            if elem[1] in tagNomi and elem[0] not in ListaPunteggiatura:
                # se Sostantivo aggiungo nella lista
                listaNomiAgg.append(elem)
            if elem[1] in tagAggettivi and elem[0] not in ListaPunteggiatura:
                # se Aggettivo aggiungo nella lista
                listaNomiAgg.append(elem)

    return listaNomiAgg


# funzione che crea lista di bigrammi aggettivo-sostantivo partendo da lista di nomi e aggettivi
def creaBigrAggSost(listaNomiAgg):
    # liste POS tags
    tagNomi = ["NN", "NNS", "NNP", "NNPS"]
    tagAggettivi = ["JJ", "JJR", "JJS"]
    # creo lista bigrammi
    listaBigrammi = nltk.bigrams(listaNomiAgg)
    # creo nuova lista bigrammi con i tag POS
    listaBigrNew = []
    # con il ciclo For assegno che se primo token di bigramma e' un Aggettivo e se secondo e' Sostantivo
    for ((tok1, tag1), (tok2, tag2)) in listaBigrammi:
        if tag1 in tagAggettivi:
            if tag2 in tagNomi:
                # aggiungo alla lista dei bigrammi
                bigramma = (tok1, tok2)
                listaBigrNew.append(bigramma)

    return listaBigrNew


# funzione che calcola probabilità condizionata di ogni bigramma sost-agg della lista, e lo uso per calcolare la prob congiunta
def probCongiunta(listaAggSost, tokensList):
    probCongiunta = []
    for ((agg, nome), freq) in listaAggSost:
        # calcolo probabilità condizionata
        probCondiz = freq * 1.0 / tokensList.count(agg) * 1.0
        # calcolo probabilità aggettivo(prima parola del bigramma)
        probAgg = tokensList.count(agg) * 1.0 / len(tokensList) * 1.0
        # calcolo probabilità congiunta
        probCong = probAgg * probCondiz
        # aggiungo nuovo elemento nella lista, con bigramma e probabilità congiunta
        bigramma = ((agg, nome), probCong)
        probCongiunta.append(bigramma)

    return probCongiunta


# funzione che Calcola forza associativa max (LMI) per ogni bigramma
# formula LMI = log2( f(a,b) * C / f(a) * f(b) )
def forzaAssociativaMax(listaAggSost, tokensList):
    # crea lista LMI di bigrammi
    listaBigrLMI = []
    for ((agg, nome), freq) in listaAggSost:
        # il ciclo for scorre solo dal lista di aggettivi e sostantivi 
        # multiplico la frequenza del bigramma e il totale di tokens nel corpus
        part1 = freq * len(tokensList)
        # frequenza della prima parola del bigramma che e' aggettivo
        fAgg = tokensList.count(agg)
        # frequenza seconda parola bigramma che e' sostantivo
        fSost = tokensList.count(nome)
        # secondo elemento della formula
        # multiplico le frequenze delle singole parole
        part2 = fAgg * fSost
        # Local Mutual Information
        LMI = freq * (math.log((part1 * 1.0 / part2 * 1.0), 2))
        # aggiungo nuovo elemento nella lista, con bigramma e LMI
        bigramma = ((agg, nome), LMI)
        listaBigrLMI.append(bigramma)
    # ordino la lista per LMI decrescenti
    listaOrdLMI = sorted(listaBigrLMI, key=lambda a: -a[1], reverse=False)

    return listaOrdLMI

# funzione che calcolo i nomi prorpi di persone con Entita Nominate
def LePersone(frasi):
    #lista dei nomi di persone
    EntitaNomePersona = []
    #scorro le frasi del testo
    for frase in frasi:
        #divido le frase ai tokens
        tokens = nltk.word_tokenize(frase)
        #lista di il POS Tag dei token 
        tokenPOS = nltk.pos_tag(tokens)
        #calcolo Entita Nominate
        namedEntity = nltk.ne_chunk(tokenPOS)
        #scorro i nodi con ciclo For
        for nodo in namedEntity:
            NE = ''
            #se nodo è un nodo intermedio o una foglia
            if hasattr(nodo, "label"):
                #se hanno presenza di l'etichetta del nodo, in questo caso i nomi propri di persone
                #se nodo.label ha presenza in NE lo aggiungo 
                if nodo.label() in ["PERSON"]:
                    #il ciclo For scorro le foglie del nodo e le unisco alle NE
                    for partNE in nodo.leaves():
                        NE = NE + ' '+ partNE[0]
                        EntitaNomePersona.append(NE)
    
    return EntitaNomePersona

# funzione ch calcola Entita Nominate
def EntitaNominate(frasi):
    #vuoto dictionary
    dict_NE= {}
    #scorro il ciclo for
    for frase in frasi:
        #divido frase ai tokens
        tokens = nltk.word_tokenize(frase)
        #il POS Tag dei token 
        tokenPOS = nltk.pos_tag(tokens)
        #calcolo Entita Nominate
        namedEntity = nltk.ne_chunk(tokenPOS)
        #scorro il ciclo for dei nodi
        for nodo in namedEntity:
            NE = ''
            #controllo se nodo è un nodo intermedio o una foglia
            if hasattr(nodo, "label"):
                #estraggo l'etichetta del nodo, in questo tutti label senza specificato
                if nodo.label() in dict_NE.keys():
                    #se si aggiungo
                    dict_NE[nodo.label()] += [NE]
                else:
                    #se no non aggiungo
                    dict_NE[nodo.label()] = [NE]
   
    # Stampa Entita Nominate con identificatori
    # Nome identificatore con Il numero dell'identificatore
    for k,v in dict_NE.items():
        print(k, len(v))
    print()

#funzione che mette nome di persone ordine di 15 piu' frequenti
def top15NE(Nominate_Identita):
    #lista che contiene i tipi
    list_identificatore = []
    #scorro i tipi con il ciclo for
    for tipi in Nominate_Identita:
        #aggiungo i tipi alla lista definita precedentemente
        list_identificatore.append(tipi)
        #calcolo al distribuzione di frequenza dei tipi
        distribuzioneType = nltk.FreqDist(list_identificatore)
        #calcolo top 15 nome di persone
        top15personi = distribuzioneType.most_common(15)
    return top15personi

#funzione che stampa 15 nome di Persone
def stampaType(mostra15Persone):
    #scorre mostra15Persone per stampare in ogni ciclo
    for elem in mostra15Persone:
        print ("\tIl nome proprio di persona", elem[0].encode("utf-8"), "occorre", elem[1], "volte"
        )

#funzione che calcola il nome del file 
def splitterName(file):
    nome, estensione = (file.name).split(".")  
    return nome

#funzione che calcola frequenza dei ogni token in ogni frase e lunghezza dei token in ogni frase
def estraiFrasi(tokensList, frasi):
    # frequenza dei token di corpus
    freqToken = nltk.FreqDist(tokensList)
    #Scorre il ciclo for
    for frase in frasi:
        listaFrasiTok = []
        #numeri di token in frase
        numTokInFrasi = 0
        # divido le frasi ai tokens
        fraseTok = nltk.word_tokenize(frase)
        # ogni token deve avere min frequenza > 2
        if all(freqToken[token] > 2 for token in fraseTok):
            #se ha 2 o piu frequenza allora controlli se token in frasi sono tra 6 e 25
            if len(fraseTok) > 6 and len(fraseTok) < 25 :
                #aggiungo nella lista e stampo ogni volta le frasi con numeri di token
                numTokInFrasi = len(fraseTok) + numTokInFrasi
                listaFrasiTok.append(fraseTok)
                print("Numeri di Token nella Frase:",numTokInFrasi,"   frase:",listaFrasiTok)
    return listaFrasiTok, freqToken

def main(file1, file2):
    
     # apre i "file1" e "file2", in sola lettura "r", in codifica "utf-8"
    fileInput1 = codecs.open(file1, "r", "utf-8")
    fileInput2 = codecs.open(file2, "r", "utf-8")

    #leggo i file
    raw1 = fileInput1.read()
    raw2 = fileInput2.read()

    # modulo statistico nltk per frasi
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    # metodo di lettura del file per la tokenizzazione
    frasi1 = sent_tokenizer.tokenize(raw1)
    frasi2 = sent_tokenizer.tokenize(raw2)

    #nome del file senza estensione
    nome1 = splitterName(fileInput1) 
    nome2 = splitterName(fileInput2)

    #Lista di Token e Lista di POS
    tokensList1, tokensPOS1 = tokenizePOS(frasi1)
    tokensList2, tokensPOS2 = tokenizePOS(frasi2)

    # sostantivi e aggettivi più frequenti
    sostMostFreq1, aggMostFreq1 = mostFreqAvvAgg(tokensPOS1)
    sostMostFreq2, aggMostFreq2 = mostFreqAvvAgg(tokensPOS2)

    # I primi 20 sostantivi
    sostMostFreq1 = top20FreqDecrs(sostMostFreq1)
    sostMostFreq2 = top20FreqDecrs(sostMostFreq2)


    # 20 bigrammi di token più frequenti (no punteggiatura)
    listaBigr1 = soloAggAvv(tokensPOS1)
    listaBigr2 = soloAggAvv(tokensPOS2)

    # prendo solo i primi 20 bigrammi
    lista20Bigr1 = top20FreqDecrs(listaBigr1)
    lista20Bigr2 = top20FreqDecrs(listaBigr2)

    # 10 POS più frequenti
    listaTag1 = listaTagPOS(tokensPOS1)
    listaTag2 = listaTagPOS(tokensPOS2)
    POSfreq1 = elem10PiuFreqDecresc(listaTag1)
    POSfreq2 = elem10PiuFreqDecresc(listaTag2)

    #calcolo i nomi propri di persone
    nomipersone1 = LePersone(frasi1)
    nomipersone2 = LePersone(frasi2)

    #calcolo i 15 nomi propri di persone più frequenti
    top15personi1 = top15NE(nomipersone1)
    top15personi2 = top15NE(nomipersone2)
    
    # 10 BIGRAMMI di POS più frequenti
    # lista bigrammi POS
    listaBigrPOS1 = POSbigrammi(tokensPOS1)
    listaBigrPOS2 = POSbigrammi(tokensPOS2)
    # prendo i primi 10 in ordine decrescente
    decrBigrPOS1 = elem10PiuFreqDecresc(listaBigrPOS1)
    decrBigrPOS2 = elem10PiuFreqDecresc(listaBigrPOS2)

    # 10 TRIGRAMMI di PoS più frequenti
    # costruisco lista trigrammi POS
    listaTrigrPOS1 = POStrigrammi(tokensPOS1)
    listaTrigrPOS2 = POStrigrammi(tokensPOS2)
    # prendo i primi 10 in ordine decrescente
    decrTrigrPOS1 = elem10PiuFreqDecresc(listaTrigrPOS1)
    decrTrigrPOS2 = elem10PiuFreqDecresc(listaTrigrPOS2)

    # sostantivi e aggettivi più frequenti
    avvMostFreq1, aggMostFreq1 = mostFreqAvvAgg(tokensPOS1)
    avvMostFreq2, aggMostFreq2 = mostFreqAvvAgg(tokensPOS2)
    # prendo solo i primi 20 sostantivi
    avvMostFreq1 = top20FreqDecrs(avvMostFreq1)
    avvMostFreq2 = top20FreqDecrs(avvMostFreq2)

    # prendo solo i primi 20 aggettivi
    aggMostFreq1 = top20FreqDecrs(aggMostFreq1)
    aggMostFreq2 = top20FreqDecrs(aggMostFreq2)

    # 20 bigrammi aggettivo-sostantivo (dove ogni token ha una frequenza > 2)
    # costruisce lista formata solo da aggettivi e sostantivi con frequenza maggiore di 2
    sostAgg1 = aggSost(tokensPOS1)
    sostAgg2 = aggSost(tokensPOS2)
    # costruisco lista bigrammi aggettivo-sostantivo
    bigrSostAgg1 = creaBigrAggSost(sostAgg1)
    bigrSostAgg2 = creaBigrAggSost(sostAgg2)
    # prendo solo i primi 20 in ordine decresc (con frequenza massima)
    bigr20AggSost1 = top20FreqDecrs(bigrSostAgg1)
    bigr20AggSost2 = top20FreqDecrs(bigrSostAgg2)

    #stampa 10 POS piu frequenti
    print ("\n10 PoS più frequenti in ordine di frequenza decrescente")
    print ("\nCORPUS", nome1, "\n",)
    for elem in POSfreq1:
        print (elem[0], "\t", elem[1])

    print ("\nCORPUS", nome2, "\n",)
    for elem in POSfreq2:
        print (elem[0], "\t", elem[1])

    #stampa 10 bigrammi di POS piu frequenti
    print ("\n10 bigrammi di PoS più frequenti in ordine di frequenza decrescente")
    print ("\nCORPUS", nome1, "\n",)
    for elem in decrBigrPOS1:
        print (elem[0], "\t", elem[1])

    print ("\nCORPUS", nome2, "\n",)
    for elem in decrBigrPOS2:
        print (elem[0], "\t", elem[1])

    #stampa 10 trigrammi di POS piu frequenti
    print ("\n10 trigrammi di PoS più frequenti in ordine di frequenza decrescente")
    print ("\nCORPUS", nome1, "\n",)
    for elem in decrTrigrPOS1:
        print (elem[0], "\t", elem[1])
    print ("\nCORPUS", nome2, "\n",)
    for elem in decrTrigrPOS2:
        print (elem[0], "\t", elem[1])

    #stampa 20 avverbi piu frequenti
    print ("\n20 avverbi più frequenti in ordine di frequenza decrescente:")
    print ("\nCORPUS", nome1, "\n",)
    for avverbo in avvMostFreq1:
        print ("Avverbo *",avverbo[0],"* ha frequenza di", avverbo[1])

    print ("\nCORPUS", nome2, "\n",)
    for avverbo in avvMostFreq2:
        print ("Avverbo *",avverbo[0],"* ha frequenza di", avverbo[1])

    #stampa 20 aggettivi piu frequenti
    print ("\n20 aggettivi più frequenti in ordine di frequenza decrescente:")
    print ("\nCORPUS", nome1, "\n",)
    for aggettivo in aggMostFreq1:
        print ("Aggettivo *",aggettivo[0],"* ha frequenza di",aggettivo[1])

    print ("\nCORPUS", nome2, "\n",)
    for aggettivo in aggMostFreq2:
        print ("Aggettivo *",aggettivo[0], "* ha frequenza di",aggettivo[1])

    #stampa 20 bigrammi di Aggettivo Sostantivo frequenza maggiore di 3
    print ("\n\n20 bigrammi aggettivo-sostantivo (dove ogni token ha una frequenza > 3)\n")
    print ("- Con Frequenza Massima\n")
    print ("\nCORPUS", nome1, "\n",)
    for (tok1, tok2), freq in bigr20AggSost1:
        print ((tok1, tok2), "\tFreq bigramma", freq)
        print ("\tAggettivo:", tok1, "\tFreq assoluta:", tokensList1.count(tok1))
        print ("\tNome:", tok2, "\tFreq assoluta:", tokensList1.count(tok2), "\n")

    print ("\nCORPUS", nome2, "\n",)
    for (tok1, tok2), freq in bigr20AggSost2:
        print((tok1, tok2), "\tFreq bigramma", freq)
        print ("\tAggettivo:", tok1, "\tFreq assoluta:", tokensList2.count(tok1))
        print ("\tNome:", tok2, "\tFreq assoluta:", tokensList2.count(tok2), "\n")

    #stampa 20 bigrammi di aggettivo-sostantivo con probabilità congiunta max
    print( "\n- Con Probabilità Congiunta\n")
    print ("\nCORPUS", nome1, "\n",)
    listaProbCong1 = probCongiunta(bigr20AggSost1, tokensList1)

    for elem in listaProbCong1:
        print (elem[0], "\n\tProbabilità:", elem[1], "\n")

    print ("\nCORPUS", nome2, "\n",)
    listaProbCong2 = probCongiunta(bigr20AggSost2, tokensList2)

    for elem in listaProbCong2:
        print (elem[0], "\n\tProbabilità:", elem[1], "\n")


    #stampa 20 bigrammi di aggettivo-sostantivo con forza associativa max (attraverso la LMI)
    print( "\n- Con Forza Associativa Massima (attraverso la Local Mutual Information)\n"
)    
    print ("\nCORPUS", nome1, "\n",)
    bigrLMI1 = forzaAssociativaMax(bigr20AggSost1, tokensList1)
    for elem in bigrLMI1:
        print (elem[0], "\n\tLocal Mutual Information:", elem[1], "\n")

    
    print ("\nCORPUS", nome2, "\n",)
    bigrLMI2 = forzaAssociativaMax(bigr20AggSost2, tokensList2)
    for elem in bigrLMI2:
        print (elem[0], "\n\tLocal Mutual Information:", elem[1], "\n")
    
    #stampa 15 nomi propri di persona più frequenti
    print ("-----15 NOMI PROPRI DI PERSONA PIU' FREQUENTI-----")
    print ("\nCORPUS", nome1, "\n",)
    stampaType(top15personi1)
    print ()
    print ("\nCORPUS", nome2, "\n",)
    stampaType(top15personi2)
    print ()
    
    #stampa le Entità Nominate
    print ("-----ENTITA NOMINATE-----")
    print ("\nCORPUS", nome1, "\n",)
    EntitaNominate(frasi1)
    print ("\nCORPUS", nome2, "\n",)
    EntitaNominate(frasi2)

    #stampa Frasi ha numeri di token tra 6 e 25 con frequnza di 2
    print ("-----Frasi min 6 token e max 25 token con min 2 frequenza-----")
    print ("\nCORPUS", nome1, "\n",)
    listaFrasiTok1, freqToken1 = estraiFrasi(tokensList1, frasi1) 
    print ("\nCORPUS", nome2, "\n",)
    listaFrasiTok2, freqToken2 = estraiFrasi(tokensList2, frasi2)



main(sys.argv[1], sys.argv[2])