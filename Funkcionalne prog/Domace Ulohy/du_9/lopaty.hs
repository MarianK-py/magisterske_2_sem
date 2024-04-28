import System.IO
import System.Environment
import Data.Char

lopaty1 :: [String] -> [Bool]
lopaty1 ss = [lopaty1' s 0 | s<-ss]

lopaty1' :: String -> Int -> Bool
lopaty1' [] _ = True
lopaty1' (s:ss) b | s == '(' = lopaty1' ss (b+1)
                  | (s == '[') && (b > 0) = lopaty1' ss b
                  | (s == '[') = False
                  | (s == ')') && (b > 0) = lopaty1' ss (b-1)
                  | otherwise = lopaty1' ss b

-- gramatika:
-- budeme mať neterminaly "a", "b" a "c" ("e" bude prazdne slovo)
-- pricom generovanie slova vzdy zacina neterminalom "a"
-- pravidla su taketo
-- a -> (b | e                            -> otvaranie () zatvorky
-- b -> [b] | [c]b | (bb | a)a | e        -> ked je otvorena pridavanie dalsich zatvoriek
-- c -> [c] | []c | (b | e                -> pridavanie zatvoriek ked vieme, ze je urcite otvorena
-- slova z prikladu ktore su spravne by mohli byt generovane takto:
-- a -> (b -> ([c]b -> ([]a)a -> ([])
-- a -> (b -> ([b] -> ([a)a] -> ([)]
-- a -> (b -> (a)a -> ()(b -> ()([c]b -> ()([]

lopaty2 :: [String] -> [Bool]
lopaty2 ss = [lopaty2' s 0 | s<-ss]

lopaty2' :: String -> Int -> Bool
lopaty2' s i | (s == "") && (i == 0) = True
             | s == "" = False
             | head s == '(' = if snd (removeZatvor "" (tail s) ')')
                               then lopaty2' (fst (removeZatvor "" (tail s) ')')) (i+1)
                               else False
             | head s == '[' = if snd (removeZatvor "" (tail s) ']')
                               then lopaty2' (fst (removeZatvor "" (tail s) ']')) (i-1)
                               else False
             | otherwise = False

removeZatvor :: String -> String -> Char -> (String, Bool)
removeZatvor ss1 [] ch = ("", False)
removeZatvor ss1 (s:ss2) ch | s == ch = (ss1++ss2, True)
                            | otherwise = removeZatvor (ss1++[s]) ss2 ch

-- idea parsera:
-- vseobecne: mat counter pocitajuci zatvorky ak uvidi () tak +1 ak uvidi [] tak -1,
-- slovo je validne ak vysledny counter je 0 (a podarilo sa cele sparsovat)
-- precitaj otvaraciu zatvorku (prvy znak), nasledne prechadzat zvysok
-- kym k nej nenajdes prvu zatvaraciu zatvorku,
-- dalej v zavislosti od zatvorky updatni counter
-- ostavaju nam 2 nesparsovane casi -> vnutro medzi prave sparsovanymi zatvorkami (oznacime x)
--                                  -> zvysok za ukoncovacov zatvorkou (oznacime y)
-- dalsie parsovanie zavisi ci ([)] je validne:
-- ak je, tak dalej parsujeme x++y (naprogramovana verzia)
-- ak nie je, tak dalej parsujeme samostatne x a y, vysledne countre sa scitaju

main :: IO ()        
main = do args <- getArgs
          let n = read (args!!0)
          putStrLn (show $ lopaty1 n)
