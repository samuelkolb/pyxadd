select_weighted(ID, Weights, Values, Value, Rest) :-
    sumlist(Weights,Total),
    sw(ID, Total, Weights, Values, Value, Rest).
select_weighted(ID, WeightsValues, Value, Rest) :-
    unzip(WeightsValues,Weights,Values),
    select_weighted(ID, Weights, Values, Value, Rest).
select_uniform(ID, Values, Value, Rest) :-
    length(Values, Len),
    Weight is 1/Len,
    build_list(Len, Weight, Weights),
    select_weighted(ID, Weights, Values, Value, Rest).

P::sw_p(ID,P,_,_,_).
sw(ID,PW,[W|WT],[X|XT],X,XT) :-
    W1 is W/PW,
    sw_p(ID,W1,WT,X,XT).
sw(ID,PW,[W|WT],[X|XT],Y,[X|RT]) :-
    W1 is W/PW,
    not sw_p(ID,W1,WT,X,XT),
    PW1 is PW-W,
    sw(ID,PW1,WT,XT,Y,RT).

build_list(0,_,[]).
build_list(Len,Val,[Val|L]) :-
    Len > 0,
    Len1 is Len-1,
    build_list(Len1,Val,L).

unzip([],[],[]).
unzip([(X,Y)|T],[X|R],[Y|S]) :-
    unzip(T,R,S).

sumlist(L,S) :- sumlist(L,0.0,S).
sumlist([],S,S).
sumlist([X|T],A,S) :-
    A1 is A+X,
    sumlist(T,A1,S).

member(X,[X|_]).
member(X,[_|T]) :-
    member(X,T).
