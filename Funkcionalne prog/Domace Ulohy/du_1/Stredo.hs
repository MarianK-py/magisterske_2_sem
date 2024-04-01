module Stredo where

-- Marian Kravec

-- priklad takehoto slova: SOS (medzinar. signal nebezpecenstva vysielany najma z lodi)
-- https://slovnik.aktuality.sk/pravopis/kratky-slovnik/?q=SOS
-- nie najlepsie slovo... ale... aj tak som sa rozhodol ho tu napisat

stredoCislo :: Integer -> Bool
stredoCislo n = stredoSlovo (show n)

stredoSlovo :: String -> Bool
stredoSlovo [] = True
stredoSlovo [s] = testChars s s
stredoSlovo (s:ss) | (testChars s (last ss)) && stredoSlovo (init ss) = True
                   | otherwise = False


testChars :: Char -> Char -> Bool
testChars c1 c2 | c1 == '0' && c2 == '0' = True
                | c1 == '1' && c2 == '1' = True
                | c1 == '8' && c2 == '8' = True
                | c1 == '6' && c2 == '9' = True
                | c1 == '9' && c2 == '6' = True
                | c1 == 'M' && c2 == 'W' = True
                | c1 == 'W' && c2 == 'M' = True
                | c1 == 'O' && c2 == 'O' = True
                | c1 == 'I' && c2 == 'I' = True
                | c1 == 'S' && c2 == 'S' = True
                | c1 == 'Z' && c2 == 'Z' = True
                | c1 == 'X' && c2 == 'X' = True
                | c1 == 'H' && c2 == 'H' = True
                | c1 == 'N' && c2 == 'N' = True
                | otherwise = False