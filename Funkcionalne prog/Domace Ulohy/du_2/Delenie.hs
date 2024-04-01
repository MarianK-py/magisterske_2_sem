module Delenie where
import Data.Char

-- Marian Kravec


-- asi to nie je uloha na rekurziu ale... nenapada mi rychlo ako to spravit cez list comprehension
splitWords  :: (Char -> Bool) -> String -> [String]
splitWords delim s = let
                       stringSplitter done curW [] | not $ null curW = done++[curW]
                                                   | otherwise = done
                       stringSplitter done curW (t:todo) | (delim t) && (not $ null curW) = stringSplitter (done++[curW]) [] todo
                                                         | (delim t) = stringSplitter done [] todo
                                                         | otherwise = stringSplitter done (curW++[t]) todo
                     in
                       stringSplitter [] [] s



whiteSpace  :: Char -> Bool
whiteSpace ch = ch == ' '        -- oddeľovač je len medzera
-- whiteSpace ch = elem ch ",.; "  -- oddeľovač je niektorý z vymenovaných znakov
-- whiteSpace ch = ord ch < ord 'A'   -- oddeľovač je všetko < 'A'
