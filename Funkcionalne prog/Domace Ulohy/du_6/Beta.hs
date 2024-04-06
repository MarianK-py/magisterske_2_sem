module Beta where

import Data.Char
import Terms

instance Show LExp where
  --show :: LExp -> String    -- definujte vlastné show
  show (ID x) = x
  show (APP m n) = "("++(show m)++" "++(show n)++")"
  show (LAMBDA x b) = "\\"++x++"."++(show b)
  show (CON s) = "("++s++")"
  show (CN x) = show x

-- \\n.\f.\x.(f ((n f) x))
a = (LAMBDA "n" (LAMBDA "f" (LAMBDA "x" (APP (ID "f") (APP (APP (ID "n") (ID "f")) (ID ("x")))))))

-- \\x.\y.((x y) \x.\y.y)
b = (LAMBDA "x" (LAMBDA "y" (APP (APP (ID "x") (ID "y")) (LAMBDA "x" (LAMBDA "y" (ID "y"))))))

k = (LAMBDA "x" (LAMBDA "y" (ID "x")))

s = (LAMBDA "x" (LAMBDA "y" (LAMBDA "z" (APP (APP (ID "x") (ID "z")) (APP (ID "y") (ID "z"))))))

izero = parse "\\f.\\x.x"
omega = parse "\\x.(x x)"
isucc  = parse "\\n.\\f.\\x.(f ((n f) x))"
iplus =  parse "\\m.\\n.\\f.\\x.((m f) ((n f) x))"
itimes = parse "\\m.\\n.\\f.\\x.((m (n f)) x)"
ipower = parse "\\m.\\n.(n m)"

--sucet = \s -> \n -> (if (= n 0) 0 (+ n (s (- n 1))))

sucet =       LAMBDA "s"
                (LAMBDA "n"
                  (APP
                   (APP
                      (APP
                         (CON "IF")
                            (APP (APP (CON "=") (ID "n")) (CN 0))   -- condition
                      )
                      (CN 0) -- then
                    )
                    (APP (APP (CON "+")    -- else
                              (ID "n"))
                              (APP (ID "s")
                                (APP (APP (CON "-") (ID "n")) (CN 1)
                                )
                             )
                    )
                   )
                )



--sucin = \s -> \n -> (COND (= n 1) 1 (* n (s (- n 1))))
sucin =       LAMBDA "s"
                (LAMBDA "n"
                  (APP
                   (APP
                      (APP
                         (CON "IF")
                            (APP (APP (CON "=") (ID "n")) (CN 1))   -- condition
                      )
                      (CN 1) -- then
                    )
                    (APP (APP (CON "*")    -- else
                              (ID "n"))
                              (APP (ID "s")
                                (APP (APP (CON "-") (ID "n")) (CN 1)
                                )
                             )
                    )
                   )
                )

ione =    (APP isucc izero)
itwo =    (APP isucc (APP isucc izero))
ifour =   (APP isucc (APP isucc (APP isucc (APP isucc izero))))
ieight =  (APP isucc (APP isucc (APP isucc (APP isucc (APP isucc (APP isucc (APP isucc (APP isucc izero))))))))
ithree =  (APP (APP iplus itwo) ione)
inine =   (APP (APP itimes ithree) ithree)
isixteen = (APP (APP ipower itwo) ifour)

isTWO = (APP (APP ipower itwo) ione)

y = LAMBDA "h"

       (APP (LAMBDA "x" (APP (ID "h") (APP (ID "x") (ID "x"))))

           (LAMBDA "x" (APP (ID "h") (APP (ID "x") (ID "x")))))

o = (LAMBDA "x" (APP (ID "x") (ID "x")))

parse' :: String -> (LExp, String)
parse' (x:xs)
    | isAlpha x = ((ID [x]), xs)
    | x == '(' = let (e1, _:res2) = parse' xs in
                 let (e2, _:res4) = parse' res2 in
                 ((APP e1 e2), res4)
    | x == '\\' = let (v:_:ys) = xs in
                  let (e, res) = parse' ys in
                  ((LAMBDA [v] e), res)
    | otherwise = error "Nieco zle sa stalo"

parse :: String -> LExp
parse = fst . parse'


-- `subterms l` vráti zoznam všetkých podtermov termu `l`
subterms :: LExp -> [LExp]
subterms (ID x) = [(ID x)]
subterms (APP m n) = (APP m n):(subterms m)++(subterms n)
subterms (LAMBDA x b) = (LAMBDA x b):(subterms b)
subterms l = []

-- `free l` vráti zoznam všetkých premenných, ktoré majú voľný výskyt v terme `l`
free :: LExp -> [Var]
free (ID x) = [x]
free (APP m n) = (free m)++(free n)
free (LAMBDA x b) = removeVar x (free b)
free l = []

removeVar :: Var -> [Var] -> [Var]
removeVar x xs = [y | y <- xs, x /= y]

-- `bound l` vráti zoznam všetkých premenných, ktoré majú viazaný výskyt v terme `l`
bound :: LExp -> [Var]
bound (ID x) = []
bound (APP m n) = (bound m)++(bound n)
bound (LAMBDA x b) = x:(bound b)
bound l = []

-- `substitute v k l` substituuje všetky voľné výskyty `v` v terme `l` za `k`
substitute :: Var -> LExp -> LExp -> LExp
substitute v k (ID x) | x == v = k
                      | otherwise = (ID x)
substitute v k (APP m n) = (APP (substitute v k m) (substitute v k n))
substitute v k (LAMBDA x b) | x == v = (LAMBDA x b)
                            | (any (==v) (free b)) && (any (==x) (free k)) = (LAMBDA z (substitute v k (substitute x (ID z) b)))
                            | otherwise = (LAMBDA x (substitute v k b))
                            where
                              z = [[y]| y <- ['a'..'z'], all (/=[y]) ((free (LAMBDA x b))++(free k))]!!0
substitute v k (CON x) = (CON x)
substitute v k (CN x) = (CN x)

hasReduction :: LExp -> Bool
hasReduction (APP (APP (CON "+") (CN i)) (CN j)) = True
hasReduction (APP (APP (CON "-") (CN i)) (CN j)) = True
hasReduction (APP (APP (CON "*") (CN i)) (CN j)) = True
hasReduction (APP (APP (CON "=") (CN x1)) (CN x2)) = True
hasReduction (APP (APP (APP (CON "IF") (CON v)) y) n) = True
hasReduction (ID x) = False
hasReduction (LAMBDA x b) = hasReduction b
hasReduction (APP (LAMBDA x b) n) = True
hasReduction (APP m n) = (hasReduction m) || (hasReduction n)
hasReduction (CON x) = False
hasReduction (CN x) = False

-- `oneStepBetaReduce l` spraví nejaké beta redukcie v `l`
-- podľa Vami zvolenej stratégie
oneStepBetaReduce :: LExp -> LExp
oneStepBetaReduce (APP (APP (CON "+") (CN i)) (CN j)) = CN (i+j)
oneStepBetaReduce (APP (APP (CON "-") (CN i)) (CN j)) = CN (i-j)
oneStepBetaReduce (APP (APP (CON "*") (CN i)) (CN j)) = CN (i*j)
oneStepBetaReduce (APP (APP (CON "=") (CN x1)) (CN x2)) | x1 == x2 = CON "TRUE"
                                                        | otherwise = CON "FALSE"
oneStepBetaReduce (APP (APP (APP (CON "IF") (CON v)) y) n) | v == "TRUE" = y
                                                           | v == "FALSE" = n
oneStepBetaReduce (ID x) = (ID x)
oneStepBetaReduce (LAMBDA x b) = (LAMBDA x (oneStepBetaReduce b))
oneStepBetaReduce (APP (LAMBDA x b) n) = substitute x n b
oneStepBetaReduce (APP m n) | (hasReduction m) && (hasReduction n) = (APP (oneStepBetaReduce m) (oneStepBetaReduce n))
                            | hasReduction m = (APP (oneStepBetaReduce m) n)
                            | hasReduction n = (APP m (oneStepBetaReduce n))
                            | otherwise = (APP m n)


-- `normalForm l` iteruje `oneStepBetaReduce` na `l`, kým sa mení
normalForm :: LExp -> LExp
normalForm l | hasReduction l = normalForm (oneStepBetaReduce l)
             | otherwise = l


alphaEq :: LExp -> LExp -> Bool
alphaEq (ID x) (ID y) | y == x = True
                      | otherwise = False
alphaEq (APP m n) (APP o p) = (alphaEq m o) && (alphaEq n p)
alphaEq (LAMBDA x a) (LAMBDA y b) | x == y = alphaEq a b
                                  | otherwise = alphaEq subX subY
                                    where
                                      z = [[w]| w <- ['a'..'z'], all (/=[w]) ((free (LAMBDA x a))++(free (LAMBDA y b)))]!!0
                                      subX = substitute x (ID z) a
                                      subY = substitute y (ID z) b
alphaEq _ _ = False

