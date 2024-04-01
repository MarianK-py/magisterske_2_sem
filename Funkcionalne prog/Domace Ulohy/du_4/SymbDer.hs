module SymbDer where


-- Marian Kravec

data Exp = ICon Int
  | Var String
  | Add Exp Exp
  | Sub Exp Exp
  | UMin Exp
  | Mul Exp Exp
  | Div Exp Exp
  | Pwr Exp Exp
  | Ln Exp   deriving (Eq, Ord, Read)


derive :: Exp -> String -> Exp
derive (ICon _) _ = ICon 0
derive (Var x) y | x == y = ICon 1
                        | otherwise = ICon 0
derive (Add e1 e2) x = urobAdd (derive e1 x) (derive e2 x)
derive (Sub e1 e2) x = urobSub (derive e1 x) (derive e2 x)
derive (UMin e) x = UMin (derive e x)
derive (Mul e1 e2) x = urobAdd (urobMul e1 (derive e2 x)) (urobMul e2 (derive e1 x))
derive (Div e1 e2) x = urobDiv (urobSub (urobMul (derive e1 x) e2) (urobMul e1 (derive e2 x))) (urobPwr e2 (ICon 2))
derive (Pwr e1 e2) x = urobMul (urobPwr e1 e2) (urobAdd (urobMul (derive e2 x) (urobLn e1)) (urobMul (urobDiv (e2) (e1)) (derive e1 x)))
derive (Ln e) x = urobMul (derive e x) (urobDiv (ICon 1) (e))

urobAdd :: Exp -> Exp -> Exp
urobAdd (ICon x) (ICon y) = ICon (x+y)
urobAdd e1 e2
  | e1 == (ICon 0) = e2
  | e2 == (ICon 0) = e1
  | otherwise = Add e1 e2

urobSub :: Exp -> Exp -> Exp
urobSub (ICon x) (ICon y) = ICon (x-y)
urobSub e1 e2
  | e1 == (ICon 0) = UMin e2
  | e2 == (ICon 0) = e1
  | e1 == e2 = ICon 0
  | otherwise = Sub e1 e2

------------------
-- Riesenie:

urobMul :: Exp -> Exp -> Exp
urobMul (ICon x) (ICon y) = ICon (x*y)
urobMul e1 e2 | e1 == (ICon 0) = (ICon 0)
              | e2 == (ICon 0) = (ICon 0)
              | e1 == (ICon 1) = e2
              | e2 == (ICon 1) = e1
              | e1 == e2 = urobPwr e1 (ICon 2)
              | otherwise = Mul e1 e2

urobDiv :: Exp -> Exp -> Exp
urobDiv (ICon x) (ICon y) = ICon (x `div` y)
urobDiv e1 e2 | e2 == (ICon 1) = e1
              | e1 == e2 = ICon 1
              | otherwise = Div e1 e2

urobPwr :: Exp -> Exp -> Exp
urobPwr (ICon x) (ICon y) = ICon (x^y)
urobPwr e1 e2 | e2 == (ICon 1) = e1
              | otherwise = Pwr e1 e2

urobLn :: Exp -> Exp
urobLn (ICon x) = ICon (floor $ log $ fromIntegral x)
urobLn e = Ln e

-----------------
instance Show Exp where
--show :: Exp -> String
  show (ICon x) = show x
  show (Var x) = x
  --show (Add e1 e2) = "(" ++ expToStr e1 ++ " + " ++ expToStr e2 ++ ")"
  --show (Sub e1 e2) = "(" ++ expToStr e1 ++ " - " ++ expToStr e2 ++ ")"
  show (Add e1 e2) = show e1 ++ " + " ++ show e2
  show (Sub e1 e2) = show e1 ++ " - (" ++ show e2 ++ ")"
  show (UMin e) = "-(" ++ show e ++ ")"
  show (Mul e1 e2) = "(" ++ show e1 ++ ")*(" ++ show e2 ++ ")"
  show (Div e1 e2) = "(" ++ show e1 ++ ")/(" ++ show e2 ++ ")"
  show (Pwr e1 e2) = "(" ++ show e1 ++ ")^(" ++ show e2 ++ ")"
  show (Ln e) =  "ln(" ++ show e ++ ")"

d v p = putStrLn (show (derive v p))

{-
vx = Var "x"
e1 = Pwr vx vx
d e1 "x" môže vrátit (x^x)*(Ln(x) + 1)
d (derive e11 "x") "x" môže vrátiť  (x^x)*((Ln(x) + 1)^2) + x^(x + -1)
d (derive (derive e11 "x") "x") "x" môže vrátiť  (x^x)*((Ln(x) + 1)^3) + 2*((x^(x + -1))*(Ln(x) + 1)) + (x^(x + -1))*(Ln(x) + (x + -1)/x)

e2 = Pwr vx (ICon 2)
d e2 "x" očakávaný 2*x, reálne kvôli vzorcu (2^x)(2/x) čo je tomu ekvivalentne

e3 = Mul (Pwr vx (ICon 2)) (Pwr vx (ICon 3))
d e3 "x" v postate x^5 čiže očakávame 5*x^4 reálne ((x)^(2))*(((x)^(3))*((3)/(x))) + ((x)^(3))*(((x)^(2))*((2)/(x)))
         čo keď začneme upravovat dostneme (x^2)*(3*x^2)+(x^3)*(2*x^1) -> 3*x^4 + 2*x^4 -> 5*x^4

e4 = Ln (Pwr (ICon 9) vx) očakávame ln(9) čiže 2 dostaneme... (((9)^(x))*(2))*((1)/((9)^(x))) -> 2
-}
vx = Var "x"
e4 = Ln (Pwr (ICon 9) vx)