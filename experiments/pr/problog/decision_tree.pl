:- consult(select).

% Expects definitions of:
%  maxdepth/1
%  features/1
%  classes/1

uniform(Low,High) :: select_threshold(N,Low,High).

feature_range(F, Range) :-
    features(Features),
    member((F,Lo,Ho),Features),
    Range is Ho-Lo.

% Remove features whose domain have become too small.
filter_features((F,L,H),R,R) :-
    feature_range(F,Range),
    H-L < 0.01*Range.
filter_features((F,L,H),R,[(F,L,H)|R]) :-
    feature_range(F,Range),
    H-L >= 0.01*Range.

% Stopping criterion
leaf(_,_,D) :- maxdepth(D).
leaf(_,[],_).
0.5::leaf(N,_,D) :- D > 0.

node(Nin,Nout,Features,MaxDepth,Classes,RClasses,class(C)) :-
    leaf(Nin,Features,MaxDepth),
    select_uniform(Nin,Classes,C,RClasses),
    Nout is Nin + 1.

node(Nin,Nout,Features,MaxDepth,Classes,Classes,tree((Feat,Value),CTrue,CFalse)) :-
    not leaf(Nin,Features,MaxDepth),
    select_uniform(Nin,Features,(Feat,Low,High),Rest),
    value(select_threshold(Nin,Low,High),Value),
    filter_features((Feat,Low,Value),Rest,Feat1),
    filter_features((Feat,Value,High),Rest,Feat2),
    MD is MaxDepth + 1,
    classes(CL),
    gen_tree(Nin,N1,Feat1,MD,CL,RCL,CTrue),
    gen_tree(N1,Nout,Feat2,MD,RCL,_,CFalse).

gen_tree(N,N1,Features,MaxDepth,Classes,RClasses,Node) :-
    node(N,N1,Features,MaxDepth,Classes,RClasses,Node).

generate_tree(Tree) :-
    features(Features),
    classes(Classes),
    gen_tree(0,_,Features,0,Classes,_,Tree).

% Use tree for classification
get_value([(F,V)|_],F,V).
get_value([_|T],F,V) :-
    get_value(T,F,V).

classify(_,class(C),C).
classify(Data,tree((Feat,Threshold),CT,CF),C) :-
    get_value(Data,Feat,Val),
    Val < Threshold,
    classify(Data,CT,C).
classify(Data,tree((Feat,Threshold),CT,CF),C) :-
    get_value(Data,Feat,Val),
    Val >= Threshold,
    classify(Data,CF,C).
