import {
  getBlacksmithShopSellPriceById as getBlacksmithShopSellPriceByIdTs,
  getBlacksmithShopTradeItemPriceById
} from './blacksmithShop'
import { evaluateLuaNumber, type LoadLuaDataModule } from './luaRuleEvaluation'

// 대장장이 매도가 규칙(매수가 → 매도가 = 절반, 최소 1)을 Lua 코드로 옮긴 것.
// 매수가는 아이템 데이터 lookup(TS)로 구하고, 가격 "공식"만 Lua 가 계산한다(BLACKSMITH_SALE_RATIO=0.5).
export const BLACKSMITH_SHOP_LUA = `
function blacksmith_sell_price(buy_price)
  return math.max(1, math.floor(buy_price * 0.5))
end
`

export const createLuaBlacksmithPricing = (loadDataModule: LoadLuaDataModule) => ({
  getSellPriceById: (itemId: string): number | undefined => {
    const buyPrice = getBlacksmithShopTradeItemPriceById(itemId)
    if (buyPrice === undefined) {
      return undefined
    }
    return evaluateLuaNumber(
      loadDataModule,
      BLACKSMITH_SHOP_LUA,
      `blacksmith_sell_price(${buyPrice})`,
      getBlacksmithShopSellPriceByIdTs(itemId) ?? 0
    )
  }
})
