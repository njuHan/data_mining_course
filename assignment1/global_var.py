# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 16:32:17 2016

@author: han
"""

ALL_FILE_NUM = 0  
DOC_NAME_LIST = []
WORD_LIST = {}
CSR_MATRIX = []
TFIDF_MATRIX = []

STOP_WORD_LIST = ["a","b","c","d","e","f","g","h","i","j","k","l",
		"m","n","o","p","q","r","s","t","u","v","w","x","y","z",
		"us","can", "able", "www","can't","cannot",
		"about","above","after","again","against",
		"all","am","an","and","any","are","aren't","as","at","be",
		"because","been","before","being","below","between","both",
		"but","by","can't","cannot","could","couldn't","did","didn't",
		"do","does","doesn't","doing","don't","down","during","each",
		"few","for","from","further","had","hadn't","has","hasn't",
		"have","haven't","having","he","he'd","he'll","he's","her","here",
		"here's","hers","herself","him","himself","his","how","how's","i",
		"i'd","i'll","i'm","i've","if","in","into","is","isn't","it","it's",
		"its","itself","let's","me","more","most","mustn't","my","myself",
		"no","nor","not","of","off","on","once","only","or","other","ought",
		"our","ours","ourselves","out","over","own","same","shan't","she",
		"she'd","she'll","she's","should","shouldn't","so","some","such",
		"than","that","that's","the","their","theirs","them","themselves",
		"then","there","there's","these","they","they'd","they'll","they're",
		"they've","this","those","through","to","too","under","until","up",
		"very","was","wasn't","we","we'd","we'll","we're","we've","were",
		"weren't","what","what's","when","when's","where","where's","which",
		"while","who","who's","whom","why","why's","with","won't","would",
		"wouldn't","you","you'd","you'll","you're","you've","your","yours",
		"yourself","yourselves","abst","accordance","according","accordingly",
		"across","act","actually","added","adj","affected","affecting","affects",
		"afterwards","ah","almost","alone","along","already","also","although",
		"always","among","amongst","announce","another","anybody","anyhow","anymore",
		"anyone","anything","anyway","anyways","anywhere","apparently","approximately",
		"aren","arent","arise","around","aside","ask","asking","auth","available","away",
		"awfully","back","became","become","becomes","becoming","beforehand","begin",
		"beginning","beginnings","begins","behind","believe","beside","besides",
		"beyond","biol","brief","briefly","ca","came","cause","causes","certain",
		"certainly","co","com","come","comes","contain","containing","contains",
		"couldnt","date","different","done","downwards","due","ed","edu","effect",
		"eg","eight","eighty","either","else","elsewhere","end","ending","enough",
		"especially","et","et-al","etc","even","ever","every","everybody","everyone",
		"everything","everywhere","ex","except","far","ff","fifth","first","five","fix",
		"followed","following","follows","former","formerly","forth","found","four",
		"furthermore","gave","get","gets","getting","give","given","gives","giving",
		"go","goes","gone","got","gotten","happens","hardly","hed","hence","hereafter",
		"hereby","herein","heres","hereupon","hes","hi","hid","hither","home","howbeit",
		"however","hundred","id","ie","im","immediate","immediately","importance",
		"important","inc","indeed","index","information","instead","invention","inward",
		"itd","it'll","just","keep","keeps","kept","kg","km","know","known","knows",
		"largely","last","lately","later","latter","latterly","least","less","lest",
		"let","lets","like","liked","likely","line","little","'ll","look","looking",
		"looks","ltd","made","mainly","make","makes","many","may","maybe","mean","means",
		"meantime","meanwhile","merely","mg","might","million","miss","ml","moreover",
		"mostly","mr","mrs","much","mug","must","na","name","namely","nay","nd","near",
		"nearly","necessarily","necessary","need","needs","neither","never","nevertheless",
		"new","next","nine","ninety","nobody","non","none","nonetheless","noone","normally",
		"nos","noted","nothing","now","nowhere","obtain","obtained","obviously","often","oh",
		"ok","okay","old","omitted","one","ones","onto","ord","others","otherwise","outside",
		"overall","owing","page","pages","part","particular","particularly","past","per",
		"perhaps","placed","please","plus","poorly","possible","possibly","potentially",
		"pp","predominantly","present","previously","primarily","probably","promptly",
		"proud","provides","put","que","quickly","quite","qv","ran","rather","rd","re",
		"readily","really","recent","recently","ref","refs","regarding","regardless",
		"regards","related","relatively","research","respectively","resulted","resulting",
		"results","right","run","said","saw","say","saying","says","sec","section","see",
		"seeing","seem","seemed","seeming","seems","seen","self","selves","sent","seven",
		"several","shall","shed","shes","show","showed","shown","showns","shows","significant",
		"significantly","similar","similarly","since","six","slightly","somebody","somehow",
		"someone","somethan","something","sometime","sometimes","somewhat","somewhere","soon",
		"sorry","specifically","specified","specify","specifying","still","stop","strongly",
		"sub","substantially","successfully","sufficiently","suggest","sup","sure","take","taken",
		"taking","tell","tends","th","thank","thanks","thanx","that'll","thats","that've","thence",
		"thereafter","thereby","thered","therefore","therein","there'll","thereof","therere",
		"theres","thereto","thereupon","there've","theyd","theyre","think","thou","though",
		"thoughh","thousand","throug","throughout","thru","thus","til","tip","together","took",
		"toward","towards","tried","tries","truly","try","trying","ts","twice","two","un",
		"unfortunately","unless","unlike","unlikely","unto","upon","ups","use","used","useful",
		"usefully","usefulness","uses","using","usually","value","various","'ve","via","viz",
		"vol","vols","vs","want","wants","wasnt","way","wed","welcome","went","werent","whatever",
		"what'll","whats","whence","whenever","whereafter","whereas","whereby","wherein",
		"wheres","whereupon","wherever","whether","whim","whither","whod","whoever","whole",
		"who'll","whomever","whos","whose","widely","willing","wish","within","without","wont",
		"words","world","wouldnt","yes","yet","youd","youre","zero","http",
		]