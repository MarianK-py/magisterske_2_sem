module List where

data List a = List [a] deriving (Show)

instance Functor List where
    -- fmap :: (a -> b) -> List a -> List b
    fmap g (List xs) = List (map g xs)

-- sematanticke pravidla
-- fmap id = id -> id sa dostane na jednotlive cleny zoznamu, nemeni sa ich poradie
--                 a kedze id clena je ta ista hodnota tak aj cely zoznam ostane nezmeneny
-- fmap (p.q) = (fmap p).(fmap q) -> v oboch pripadoch na obe cleny zoznamu najprv aplikujeme q a potom p
--                                   preto v oboch pripadoch dostaneme rovnaky vysledok

instance Applicative List where
    -- pure :: a -> List a
   pure x = List [x]
   -- (<*>) :: List (a -> b) -> List a -> List b
   (List gs) <*> (List xs) = List ([g x | g <- gs, x <- xs])

-- sematanticke pravidla
-- pure id <*> v = v -> v podstate fmap id v o com vieme, ze je v
-- pure (.) <*> u <*> v <*> w = u <*> (v <*> w) -> lava strana sa vyhodnoti na: List [...(ui.vj) wk,...]
--                                                 prava strana sa vyhodnoti na: List [...ui (vj wk),...]
--                                                 čo je len iný zápis toho istého
-- pure f <*> pure x = pure (f x) -> lava strana sa vyhodnoti na: List [f x] (kedze obe zoznamy su jednoprvkove)
--                                   prava strana bude totozna
-- u <*> pure y = pure ($ y) <*> u -> lava strana sa vyhodnoti na: List [...ui y,...]
--                                    prava strana sa vyhodnoti na: List [...($ y) ui,...] co je List [...ui y,...]