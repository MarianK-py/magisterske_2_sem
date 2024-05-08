module Sheep where
import Data.Maybe
import Control.Monad

-- a sheep has its name, and maybe mother and father
data Sheep = Sheep {name::String, mother::Maybe Sheep, father::Maybe Sheep}   deriving (Eq)

instance Show Sheep where
  show s = show (name s)

-- convert a Maybe value into another monad
maybeToMonad :: (MonadPlus m) => Maybe a -> m a
maybeToMonad Nothing  = mzero
maybeToMonad (Just s) = return s

k_mother :: Int -> Sheep -> Maybe Sheep
k_mother 0 s = Just s
k_mother n s = do s_m <- (k_mother (n-1) s)
                  mother s_m

k_predecesors :: Int -> Sheep -> [Sheep]
k_predecesors 0 s = [s]
k_predecesors n s = do k <- k_predecesors (n-1) s
                       (maybeToMonad (father k))++(maybeToMonad (mother k))

---- nejake data:
adam   = Sheep "Adam"    Nothing Nothing
eve    = Sheep "Eve"     Nothing Nothing
uranus = Sheep "Uranus"  Nothing Nothing
gaea   = Sheep "Gaea"    Nothing Nothing
kronos = Sheep "Kronos"  (Just gaea) (Just uranus)
holly  = Sheep "Holly"   (Just eve) (Just adam)
roger  = Sheep "Roger"   (Just eve) (Just kronos)
molly  = Sheep "Molly"   (Just holly) (Just roger)
dolly  = Sheep "Dolly"   (Just molly) Nothing
