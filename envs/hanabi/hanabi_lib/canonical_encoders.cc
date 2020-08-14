// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "canonical_encoders.h"

namespace hanabi_learning_env {

namespace {

// Computes the product of dimensions in shape, i.e. how many individual
// pieces of data the encoded observation requires.
int FlatLength(const std::vector<int>& shape) {
  return std::accumulate(std::begin(shape), std::end(shape), 1,
                         std::multiplies<int>());
}

const HanabiHistoryItem* GetLastNonDealMove(
    const std::vector<HanabiHistoryItem>& past_moves) {
  auto it = std::find_if(
      past_moves.begin(), past_moves.end(), [](const HanabiHistoryItem& item) {
        return item.move.MoveType() != HanabiMove::Type::kDeal;
      });
  return it == past_moves.end() ? nullptr : &(*it);
}

int BitsPerCard(const HanabiGame& game) {
  return game.NumColors() * game.NumRanks();
}

// The card's one-hot index using a color-major ordering.
int CardIndex(int color, int rank, int num_ranks) {
  return color * num_ranks + rank;
}

int HandsSectionLength(const HanabiGame& game) {
  return game.NumPlayers() * game.HandSize() * BitsPerCard(game) +
         game.NumPlayers();
}

// // encode eps: [0, 1) into normalized log range
// int EncodeEps(const HanabiGame& game,
//               const HanabiObservation& obs,
//               int start_offset,
//               const std::vector<float>* eps,
//               std::vector<float>* encoding) {
//   const int num_players = game.NumPlayers();
//   const float tiny = 1e-6;
//   const float log_tiny = std::log(tiny);

//   int observing_player = obs.ObservingPlayer();
//   int code_offset = start_offset;
//   if (eps != nullptr) {
//     assert(eps->size() == game.NumPlayers());
//   }
//   for (int offset = 1; offset < game.NumPlayers(); ++offset) {
//     float player_eps = 0;
//     if (eps != nullptr) {
//       player_eps = (*eps)[(offset + observing_player) % num_players];
//     }
//     player_eps += tiny;

//     // TODO: magical number 19 to make it [0, 1)
//     float normed = (std::log(player_eps) - log_tiny) / (-log_tiny);
//     assert(normed >= 0 && normed < 1);
//     (*encoding)[code_offset] = normed;
//     ++code_offset;
//   }
//   return code_offset - start_offset;
// }

int EncodeOwnHand_(const HanabiGame& game,
                  const HanabiObservation& obs,
                  int start_offset,
                  std::vector<float>* encoding) {
  int bits_per_card = 3; // BitsPerCard(game);
  int num_ranks = game.NumRanks();
  // int hand_size = game.HandSize();

  int offset = start_offset;
  const std::vector<HanabiHand>& hands = obs.Hands();
  const int player = 0;
  const std::vector<HanabiCard>& cards = hands[player].Cards();
  // int num_cards = 0;

  const std::vector<int>& fireworks = obs.Fireworks();
  for (const HanabiCard& card : cards) {
    // Only a player's own cards can be invalid/unobserved.
    // assert(card.IsValid());
    assert(card.Color() < game.NumColors());
    assert(card.Rank() < num_ranks);
    assert(card.IsValid());
    // std::cout << offset << CardIndex(card.Color(), card.Rank(), num_ranks) << std::endl;
    // std::cout << card.Color() << ", " << card.Rank() << ", " << num_ranks << std::endl;
    auto firework = fireworks[card.Color()];
    if (card.Rank() == firework) {
      (*encoding)[offset] = 1;
    } else if (card.Rank() < firework) {
      (*encoding)[offset + 1] = 1;
    } else {
      (*encoding)[offset + 2] = 1;
    }

    offset += bits_per_card;
  }

  return offset;
}

// Enocdes cards in all other player's hands (excluding our unknown hand),
// and whether the hand is missing a card for all players (when deck is empty.)
// Each card in a hand is encoded with a one-hot representation using
// <num_colors> * <num_ranks> bits (25 bits in a standard game) per card.
// Returns the number of entries written to the encoding.
int EncodeHands(const HanabiGame& game,
                const HanabiObservation& obs,
                int start_offset,
                std::vector<float>* encoding,
                bool show_own_cards) {
  int bits_per_card = BitsPerCard(game);
  int num_ranks = game.NumRanks();
  int num_players = game.NumPlayers();
  int hand_size = game.HandSize();

  int offset = start_offset;
  const std::vector<HanabiHand>& hands = obs.Hands();
  assert(hands.size() == num_players);
  for (int player = 0; player < num_players; ++player) {
    const std::vector<HanabiCard>& cards = hands[player].Cards();
    int num_cards = 0;

    for (const HanabiCard& card : cards) {
      // Only a player's own cards can be invalid/unobserved.
      // assert(card.IsValid());
      assert(card.Color() < game.NumColors());
      assert(card.Rank() < num_ranks);
      if (player == 0) {
        if (show_own_cards) {
          assert(card.IsValid());
          // std::cout << offset << CardIndex(card.Color(), card.Rank(), num_ranks) << std::endl;
          // std::cout << card.Color() << ", " << card.Rank() << ", " << num_ranks << std::endl;
          (*encoding).at(offset + CardIndex(card.Color(), card.Rank(), num_ranks)) = 1;
        } else {
          assert(!card.IsValid());
          // (*encoding).at(offset + CardIndex(card.Color(), card.Rank(), num_ranks)) = 0;
        }
      } else {
        assert(card.IsValid());
        (*encoding).at(offset + CardIndex(card.Color(), card.Rank(), num_ranks)) = 1;
      }

      ++num_cards;
      offset += bits_per_card;
    }

    // A player's hand can have fewer cards than the initial hand size.
    // Leave the bits for the absent cards empty (adjust the offset to skip
    // bits for the missing cards).
    if (num_cards < hand_size) {
      offset += (hand_size - num_cards) * bits_per_card;
    }
  }

  // For each player, set a bit if their hand is missing a card.
  for (int player = 0; player < num_players; ++player) {
    if (hands[player].Cards().size() < game.HandSize()) {
      (*encoding)[offset + player] = 1;
    }
  }
  offset += num_players;

  assert(offset - start_offset == HandsSectionLength(game));
  return offset - start_offset;
}

int BoardSectionLength(const HanabiGame& game) {
  return game.MaxDeckSize() - game.NumPlayers() * game.HandSize() +  // deck
         game.NumColors() * game.NumRanks() +  // fireworks
         game.MaxInformationTokens() +         // info tokens
         game.MaxLifeTokens();                 // life tokens
}

// Encode the board, including:
//   - remaining deck size
//     (max_deck_size - num_players * hand_size bits; thermometer)
//   - state of the fireworks (<num_ranks> bits per color; one-hot)
//   - information tokens remaining (max_information_tokens bits; thermometer)
//   - life tokens remaining (max_life_tokens bits; thermometer)
// We note several features use a thermometer representation instead of one-hot.
// For example, life tokens could be: 000 (0), 100 (1), 110 (2), 111 (3).
// Returns the number of entries written to the encoding.
int EncodeBoard(const HanabiGame& game, const HanabiObservation& obs,
                int start_offset, std::vector<float>* encoding) {
  int num_colors = game.NumColors();
  int num_ranks = game.NumRanks();
  int num_players = game.NumPlayers();
  int hand_size = game.HandSize();
  int max_deck_size = game.MaxDeckSize();

  int offset = start_offset;
  // Encode the deck size
  for (int i = 0; i < obs.DeckSize(); ++i) {
    (*encoding)[offset + i] = 1;
  }
  // std::cout << "max_deck_size: " << max_deck_size
  //           << ", deck_size: " << obs.DeckSize() << std::endl;
  offset += (max_deck_size - hand_size * num_players);  // 40 in normal 2P game

  // fireworks
  const std::vector<int>& fireworks = obs.Fireworks();
  for (int c = 0; c < num_colors; ++c) {
    // fireworks[color] is the number of successfully played <color> cards.
    // If some were played, one-hot encode the highest (0-indexed) rank played
    if (fireworks[c] > 0) {
      (*encoding)[offset + fireworks[c] - 1] = 1;
    }
    offset += num_ranks;
  }

  // info tokens
  assert(obs.InformationTokens() >= 0);
  assert(obs.InformationTokens() <= game.MaxInformationTokens());
  for (int i = 0; i < obs.InformationTokens(); ++i) {
    (*encoding)[offset + i] = 1;
  }
  offset += game.MaxInformationTokens();

  // life tokens
  assert(obs.LifeTokens() >= 0);
  assert(obs.LifeTokens() <= game.MaxLifeTokens());
  for (int i = 0; i < obs.LifeTokens(); ++i) {
    (*encoding)[offset + i] = 1;
  }
  offset += game.MaxLifeTokens();

  assert(offset - start_offset == BoardSectionLength(game));
  return offset - start_offset;
}

int DiscardSectionLength(const HanabiGame& game) { return game.MaxDeckSize(); }

// Encode the discard pile. (max_deck_size bits)
// Encoding is in color-major ordering, as in kColorStr ("RYGWB"), with each
// color and rank using a thermometer to represent the number of cards
// discarded. For example, in a standard game, there are 3 cards of lowest rank
// (1), 1 card of highest rank (5), 2 of all else. So each color would be
// ordered like so:
//
//   LLL      H
//   1100011101
//
// This means for this color:
//   - 2 cards of the lowest rank have been discarded
//   - none of the second lowest rank have been discarded
//   - both of the third lowest rank have been discarded
//   - one of the second highest rank have been discarded
//   - the highest rank card has been discarded
// Returns the number of entries written to the encoding.
int EncodeDiscards(const HanabiGame& game, const HanabiObservation& obs,
                   int start_offset, std::vector<float>* encoding) {
  int num_colors = game.NumColors();
  int num_ranks = game.NumRanks();

  int offset = start_offset;
  std::vector<int> discard_counts(num_colors * num_ranks, 0);
  for (const HanabiCard& card : obs.DiscardPile()) {
    ++discard_counts[card.Color() * num_ranks + card.Rank()];
  }

  for (int c = 0; c < num_colors; ++c) {
    for (int r = 0; r < num_ranks; ++r) {
      int num_discarded = discard_counts[c * num_ranks + r];
      for (int i = 0; i < num_discarded; ++i) {
        (*encoding)[offset + i] = 1;
      }
      offset += game.NumberCardInstances(c, r);
    }
  }

  assert(offset - start_offset == DiscardSectionLength(game));
  return offset - start_offset;
}

// Encode the last player action (not chance's deal of cards). This encodes:
//  - Acting player index, relative to ourself (<num_players> bits; one-hot)
//  - The MoveType (4 bits; one-hot)
//  - Target player index, relative to acting player, if a reveal move
//    (<num_players> bits; one-hot)
//  - Color revealed, if a reveal color move (<num_colors> bits; one-hot)
//  - Rank revealed, if a reveal rank move (<num_ranks> bits; one-hot)
//  - Reveal outcome (<hand_size> bits; each bit is 1 if the card was hinted at)
//  - Position played/discarded (<hand_size> bits; one-hot)
//  - Card played/discarded (<num_colors> * <num_ranks> bits; one-hot)
// Returns the number of entries written to the encoding.
int EncodeLastAction_(const HanabiGame& game, const HanabiObservation& obs,
                     int start_offset, std::vector<float>* encoding) {
  int num_colors = game.NumColors();
  int num_ranks = game.NumRanks();
  int num_players = game.NumPlayers();
  int hand_size = game.HandSize();

  int offset = start_offset;
  const HanabiHistoryItem* last_move = GetLastNonDealMove(obs.LastMoves());
  if (last_move == nullptr) {
    offset += LastActionSectionLength(game);
  } else {
    HanabiMove::Type last_move_type = last_move->move.MoveType();

    // player_id
    // Note: no assertion here. At a terminal state, the last player could have
    // been me (player id 0).
    (*encoding)[offset + last_move->player] = 1;
    offset += num_players;

    // move type
    switch (last_move_type) {
      case HanabiMove::Type::kPlay:
        (*encoding)[offset] = 1;
        break;
      case HanabiMove::Type::kDiscard:
        (*encoding)[offset + 1] = 1;
        break;
      case HanabiMove::Type::kRevealColor:
        (*encoding)[offset + 2] = 1;
        break;
      case HanabiMove::Type::kRevealRank:
        (*encoding)[offset + 3] = 1;
        break;
      default:
        std::abort();
    }
    offset += 4;

    // target player (if hint action)
    if (last_move_type == HanabiMove::Type::kRevealColor ||
        last_move_type == HanabiMove::Type::kRevealRank) {
      int8_t observer_relative_target =
          (last_move->player + last_move->move.TargetOffset()) % num_players;
      (*encoding)[offset + observer_relative_target] = 1;
    }
    offset += num_players;

    // color (if hint action)
    if (last_move_type == HanabiMove::Type::kRevealColor) {
      (*encoding)[offset + last_move->move.Color()] = 1;
    }
    offset += num_colors;

    // rank (if hint action)
    if (last_move_type == HanabiMove::Type::kRevealRank) {
      (*encoding)[offset + last_move->move.Rank()] = 1;
    }
    offset += num_ranks;

    // outcome (if hinted action)
    if (last_move_type == HanabiMove::Type::kRevealColor ||
        last_move_type == HanabiMove::Type::kRevealRank) {
      for (int i = 0, mask = 1; i < hand_size; ++i, mask <<= 1) {
        if ((last_move->reveal_bitmask & mask) > 0) {
          (*encoding)[offset + i] = 1;
        }
      }
    }
    offset += hand_size;

    // position (if play or discard action)
    if (last_move_type == HanabiMove::Type::kPlay ||
        last_move_type == HanabiMove::Type::kDiscard) {
      (*encoding)[offset + last_move->move.CardIndex()] = 1;
    }
    offset += hand_size;

    // card (if play or discard action)
    if (last_move_type == HanabiMove::Type::kPlay ||
        last_move_type == HanabiMove::Type::kDiscard) {
      assert(last_move->color >= 0);
      assert(last_move->rank >= 0);
      (*encoding)[offset +
                  CardIndex(last_move->color, last_move->rank, num_ranks)] = 1;
    }
    offset += BitsPerCard(game);

    // was successful and/or added information token (if play action)
    if (last_move_type == HanabiMove::Type::kPlay) {
      if (last_move->scored) {
        (*encoding)[offset] = 1;
      }
      if (last_move->information_token) {
        (*encoding)[offset + 1] = 1;
      }
    }
    offset += 2;
  }

  assert(offset - start_offset == LastActionSectionLength(game));
  return offset - start_offset;
}

int CardKnowledgeSectionLength(const HanabiGame& game) {
  return game.NumPlayers() * game.HandSize() *
         (BitsPerCard(game) + game.NumColors() + game.NumRanks());
}

// Encode the common card knowledge.
// For each card/position in each player's hand, including the observing player,
// encode the possible cards that could be in that position and whether the
// color and rank were directly revealed by a Reveal action. Possible card
// values are in color-major order, using <num_colors> * <num_ranks> bits per
// card. For example, if you knew nothing about a card, and a player revealed
// that is was green, the knowledge would be encoded as follows.
// R    Y    G    W    B
// 0000000000111110000000000   Only green cards are possible.
// 0    0    1    0    0       Card was revealed to be green.
// 00000                       Card rank was not revealed.
//
// Similarly, if the player revealed that one of your other cards was green, you
// would know that this card could not be green, resulting in:
// R    Y    G    W    B
// 1111111111000001111111111   Any card that is not green is possible.
// 0    0    0    0    0       Card color was not revealed.
// 00000                       Card rank was not revealed.
// Uses <num_players> * <hand_size> *
// (<num_colors> * <num_ranks> + <num_colors> + <num_ranks>) bits.
// Returns the number of entries written to the encoding.
int EncodeCardKnowledge(const HanabiGame& game, const HanabiObservation& obs,
                        int start_offset, std::vector<float>* encoding) {
  int bits_per_card = BitsPerCard(game);
  int num_colors = game.NumColors();
  int num_ranks = game.NumRanks();
  int num_players = game.NumPlayers();
  int hand_size = game.HandSize();

  int offset = start_offset;
  const std::vector<HanabiHand>& hands = obs.Hands();
  assert(hands.size() == num_players);
  for (int player = 0; player < num_players; ++player) {
    const std::vector<HanabiHand::CardKnowledge>& knowledge =
        hands[player].Knowledge();
    int num_cards = 0;

    for (const HanabiHand::CardKnowledge& card_knowledge : knowledge) {
      // Add bits for plausible card.
      for (int color = 0; color < num_colors; ++color) {
        if (card_knowledge.ColorPlausible(color)) {
          for (int rank = 0; rank < num_ranks; ++rank) {
            if (card_knowledge.RankPlausible(rank)) {
              (*encoding)[offset + CardIndex(color, rank, num_ranks)] = 1;
            }
          }
        }
      }
      offset += bits_per_card;

      // Add bits for explicitly revealed colors and ranks.
      if (card_knowledge.ColorHinted()) {
        (*encoding)[offset + card_knowledge.Color()] = 1;
      }
      offset += num_colors;
      if (card_knowledge.RankHinted()) {
        (*encoding)[offset + card_knowledge.Rank()] = 1;
      }
      offset += num_ranks;

      ++num_cards;
    }

    // A player's hand can have fewer cards than the initial hand size.
    // Leave the bits for the absent cards empty (adjust the offset to skip
    // bits for the missing cards).
    if (num_cards < hand_size) {
      offset +=
          (hand_size - num_cards) * (bits_per_card + num_colors + num_ranks);
    }
  }

  assert(offset - start_offset == CardKnowledgeSectionLength(game));
  return offset - start_offset;
}

std::vector<int> ComputeCardCount(const HanabiGame& game, const HanabiObservation& obs) {
  int num_colors = game.NumColors();
  int num_ranks = game.NumRanks();

  std::vector<int> card_count(num_colors * num_ranks, 0);
  int total_count = 0;
  // full deck card count
  for (int color = 0; color < game.NumColors(); ++color) {
    for (int rank = 0; rank < game.NumRanks(); ++rank) {
      auto count = game.NumberCardInstances(color, rank);
      card_count[color * num_ranks + rank] = count;
      total_count += count;
    }
  }
  // remove discard
  for (const HanabiCard& card : obs.DiscardPile()) {
    --card_count[card.Color() * num_ranks + card.Rank()];
    --total_count;
  }
  // remove fireworks on board
  const std::vector<int>& fireworks = obs.Fireworks();
  for (int c = 0; c < num_colors; ++c) {
    // fireworks[color] is the number of successfully played <color> cards.
    // If some were played, one-hot encode the highest (0-indexed) rank played
    if (fireworks[c] > 0) {
      for (int rank = 0; rank < fireworks[c]; ++rank) {
        --card_count[c * num_ranks + rank];
        --total_count;
      }
    }
  }

  {
    // sanity check
    const std::vector<HanabiHand>& hands = obs.Hands();
    int total_hand_size = 0;
    for (const auto& hand : hands) {
      total_hand_size += hand.Cards().size();
    }
    if(total_count != obs.DeckSize() + total_hand_size) {
      std::cout << "size mismatch: " << total_count
                << " vs " << obs.DeckSize() + total_hand_size << std::endl;
      assert(false);
    }
  }
  return card_count;
}

int EncodeV0Belief_(const HanabiGame& game,
                   const HanabiObservation& obs,
                   int start_offset,
                   std::vector<float>* encoding,
                   std::vector<int>* ret_card_count=nullptr) {
  // int bits_per_card = BitsPerCard(game);
  int num_colors = game.NumColors();
  int num_ranks = game.NumRanks();
  int num_players = game.NumPlayers();
  int hand_size = game.HandSize();

  std::vector<int> card_count = ComputeCardCount(game, obs);
  if (ret_card_count != nullptr) {
    *ret_card_count = card_count;
  }

  // card knowledge
  const int len = EncodeCardKnowledge(game, obs, start_offset, encoding);
  const int player_offset = len / num_players;
  const int per_card_offset = len / hand_size / num_players;
  assert(per_card_offset == num_colors * num_ranks + num_colors + num_ranks);

  const std::vector<HanabiHand>& hands = obs.Hands();
  for (int player_id = 0; player_id < num_players; ++player_id) {
    int num_cards = hands[player_id].Cards().size();
    for (int card_idx = 0; card_idx < num_cards; ++card_idx) {
      float total = 0;
      for (int i = 0; i < num_colors * num_ranks; ++i) {
        int offset = (start_offset
                      + player_offset * player_id
                      + card_idx * per_card_offset
                      + i);
        // std::cout << offset << ", " << len << std::endl;
        assert(offset - start_offset < len);
        (*encoding)[offset] *= card_count[i];
        total += (*encoding)[offset];
      }
      if (total <= 0) {
        // const std::vector<HanabiHand>& hands = obs.Hands();
        std::cout << hands[0].Cards().size() << std::endl;
        std::cout << hands[1].Cards().size() << std::endl;
        std::cout << "total = 0 " << std::endl;
        assert(false);
      }
      for (int i = 0; i < num_colors * num_ranks; ++i) {
        int offset = (start_offset
                      + player_offset * player_id
                      + card_idx * per_card_offset
                      + i);
        (*encoding)[offset] /= total;
      }
    }
  }
  return len;
}

int EncodeV1Belief_(const HanabiGame& game,
                   const HanabiObservation& obs,
                   int start_offset,
                   std::vector<float>* encoding) {
  const int num_iters = 100;
  const float weight = 0.1;

  // int bits_per_card = BitsPerCard(game);
  int num_colors = game.NumColors();
  int num_ranks = game.NumRanks();
  int num_players = game.NumPlayers();
  int hand_size = game.HandSize();
  const std::vector<HanabiHand>& hands = obs.Hands();

  std::vector<float> card_knowledge(CardKnowledgeSectionLength(game), 0);
  int len = EncodeCardKnowledge(game, obs, 0, &card_knowledge);
  assert(len == card_knowledge.size());

  std::vector<float> v0_belief(card_knowledge);
  std::vector<int> card_count;
  len = EncodeV0Belief_(game, obs, 0, &v0_belief, &card_count);
  assert(len == card_knowledge.size());

  const int player_offset = len / num_players;
  const int per_card_offset = len / hand_size / num_players;
  assert(per_card_offset == num_colors * num_ranks + num_colors + num_ranks);

  std::vector<float> v1_belief(v0_belief);
  std::vector<float> new_v1_belief(v1_belief);
  std::vector<float> total_cards(card_count.size());

  assert(total_cards.size() == int(num_colors * num_ranks));
  for (int step = 0; step < num_iters; ++step) {
    // first compute total card remaining by excluding info from belief
    for (int i = 0; i < num_colors * num_ranks; ++i) {
      total_cards[i] = card_count[i];
      for (int player_id = 0; player_id < num_players; ++player_id) {
        int num_cards = hands[player_id].Cards().size();
        for (int card_idx = 0; card_idx < num_cards; ++card_idx) {
          int offset = player_offset * player_id + card_idx * per_card_offset + i;
          assert(offset < (int)v1_belief.size());
          total_cards[i] -= v1_belief[offset];
        }
      }
    }
    // for (auto c : total_cards) {
    //   std::cout << c << ", ";
    // }
    // std::cout << std::endl;

    // compute new belief
    for (int player_id = 0; player_id < num_players; ++player_id) {
      int num_cards = hands[player_id].Cards().size();
      for (int card_idx = 0; card_idx < num_cards; ++card_idx) {
        // float total = 0;
        int base_offset = player_offset * player_id + card_idx * per_card_offset;
        // if (player_id == 0 && card_idx == 0) {
        //   std::cout << "before norm" << std::endl;
        // }
        for (int i = 0; i < num_colors * num_ranks; ++i) {
          int offset = base_offset + i;
          assert(offset < (int)v1_belief.size());
          float p = total_cards[i] + v1_belief[offset];
          // if (player_id == 0 && card_idx == 0) {
          //   std::cout << p << ", ";
          // }
          p = std::max(p, (float)0.0);
          new_v1_belief[offset] = p * card_knowledge[offset];
          // total += new_v1_belief[offset];
        }
        // // std::cout << std::endl;
        // if (total <= 0) {
        //   // const std::vector<HanabiHand>& hands = obs.Hands();
        //   // std::cout << hands[0].Cards().size() << std::endl;
        //   // std::cout << hands[1].Cards().size() << std::endl;
        //   // std::cout << "total = 0 " << std::endl;
        //   // assert(false);
        //   total = 1;
        // }
        // // if (player_id == 0 && card_idx == 0) {
        // //   std::cout << "total: " << total << std::endl;
        // // }
        // // normalize
        // for (int i = 0; i < num_colors * num_ranks; ++i) {
        //   int offset = base_offset + i;
        //   new_v1_belief[offset] /= total;
        // }
        // // if (player_id == 0 && card_idx == 0) {
        // //   for (int i = 0; i < num_colors * num_ranks; ++i) {
        // //     int offset = base_offset + i;
        // //     std::cout << new_v1_belief[offset] << ", ";
        // //   }
        // //   std::cout << std::endl;
        // // }
      }
    }
    // interpolate & normalize
    for (int player_id = 0; player_id < num_players; ++player_id) {
      int num_cards = hands[player_id].Cards().size();
      for (int card_idx = 0; card_idx < num_cards; ++card_idx) {
        float total = 0;
        int base_offset = player_offset * player_id + card_idx * per_card_offset;
        for (int i = 0; i < num_colors * num_ranks; ++i) {
          int offset = i + base_offset;
          v1_belief[offset] = (1 - weight) * v1_belief[offset] + weight * new_v1_belief[offset];
          total += v1_belief[offset];
        }
        if (total <= 0) {
          std::cout << "total = 0 " << std::endl;
          assert(false);
        }
        for (int i = 0; i < num_colors * num_ranks; ++i) {
          int offset = i + base_offset;
          v1_belief[offset] /= total;
        }
      }
    }
  }

  for (size_t i = 0; i < v1_belief.size(); ++i) {
    (*encoding)[i + start_offset] = v1_belief[i];
  }
  return v1_belief.size();
}

}  // namespace

int LastActionSectionLength(const HanabiGame& game) {
  return game.NumPlayers() +  // player id
         4 +                  // move types (play, dis, rev col, rev rank)
         game.NumPlayers() +  // target player id (if hint action)
         game.NumColors() +   // color (if hint action)
         game.NumRanks() +    // rank (if hint action)
         game.HandSize() +    // outcome (if hint action)
         game.HandSize() +    // position (if play action)
         BitsPerCard(game) +  // card (if play or discard action)
         2;                   // play (successful, added information token)
}

std::vector<int> CanonicalObservationEncoder::Shape() const {
  int l = HandsSectionLength(*parent_game_) +
          BoardSectionLength(*parent_game_) +
          DiscardSectionLength(*parent_game_) +
          LastActionSectionLength(*parent_game_) +
          (parent_game_->ObservationType() == HanabiGame::kMinimal
               ? 0
               : CardKnowledgeSectionLength(*parent_game_));
  return {l};
}

std::vector<float> CanonicalObservationEncoder::EncodeLastAction(
    const HanabiObservation& obs) const {
  std::vector<float> encoding(LastActionSectionLength(*parent_game_), 0);
  int offset = 0;
  offset += EncodeLastAction_(*parent_game_, obs, offset, &encoding);
  assert(offset == encoding.size());
  return encoding;
}

std::vector<float> ExtractBelief(const std::vector<float>& encoding,
                                 const HanabiGame& game) {
  int bits_per_card = BitsPerCard(game);
  int num_colors = game.NumColors();
  int num_ranks = game.NumRanks();
  int num_players = game.NumPlayers();
  int hand_size = game.HandSize();
  int encoding_sector_len = bits_per_card + num_colors + num_ranks;
  assert(encoding_sector_len * hand_size * num_players == (int)encoding.size());

  std::vector<float> belief(num_players * hand_size * bits_per_card);
  for (int i = 0; i < num_players; ++i) {
    for (int j = 0; j < hand_size; ++j) {
      for (int k = 0; k < bits_per_card; ++k) {
        int belief_offset = (i * hand_size + j) * bits_per_card + k;
        int encoding_offset = (i * hand_size + j) * encoding_sector_len + k;
        belief[belief_offset] = encoding[encoding_offset];
      }
    }
  }
  return belief;
}

std::vector<float> CanonicalObservationEncoder::EncodeV0Belief(
    const HanabiObservation& obs) const {
  std::vector<float> encoding(CardKnowledgeSectionLength(*parent_game_), 0);
  int len = EncodeV0Belief_(*parent_game_, obs, 0, &encoding);
  assert(len == (int)encoding.size());
  auto belief = ExtractBelief(encoding, *parent_game_);
  return belief;
}

std::vector<float> CanonicalObservationEncoder::EncodeV1Belief(
    const HanabiObservation& obs) const {
  std::vector<float> encoding(CardKnowledgeSectionLength(*parent_game_), 0);
  int len = EncodeV1Belief_(*parent_game_, obs, 0, &encoding);
  assert(len == (int)encoding.size());
  auto belief = ExtractBelief(encoding, *parent_game_);
  return belief;
}

std::vector<float> CanonicalObservationEncoder::EncodeHandMask(
    const HanabiObservation& obs) const {
  std::vector<float> encoding(CardKnowledgeSectionLength(*parent_game_), 0);
  // const int len = EncodeCardKnowledge(game, obs, start_offset, encoding);
  EncodeCardKnowledge(*parent_game_, obs, 0, &encoding);
  auto hm = ExtractBelief(encoding, *parent_game_);
  return hm;
}

std::vector<float> CanonicalObservationEncoder::EncodeCardCount(
    const HanabiObservation& obs) const {
  std::vector<float> encoding;
  auto cc = ComputeCardCount(*parent_game_, obs);
  for (size_t i = 0; i < cc.size(); ++i) {
    encoding.push_back((float)cc[i]);
  }
  return encoding;
}

std::vector<float> CanonicalObservationEncoder::EncodeOwnHand(
    const HanabiObservation& obs) const {
  // int len = parent_game_->HandSize() * BitsPerCard(*parent_game_);
  // hard code 5 cards, empty slot will be all zero
  int len = 5 * 3;
  // std::cout << "len: " << len << std::endl;
  std::vector<float> encoding(len, 0);
  int offset = EncodeOwnHand_(*parent_game_, obs, 0, &encoding);
  assert(offset <= len);
  return encoding;
}

std::vector<float> CanonicalObservationEncoder::Encode(
    const HanabiObservation& obs,
    bool show_own_cards) const {
  // Make an empty bit string of the proper size.
  std::vector<float> encoding(FlatLength(Shape()), 0);
  // std::cout << "encoding shape: " << encoding.size() << std::endl;

  // This offset is an index to the start of each section of the bit vector.
  // It is incremented at the end of each section.
  int offset = 0;

  offset += EncodeHands(*parent_game_, obs, offset, &encoding, show_own_cards);
  offset += EncodeBoard(*parent_game_, obs, offset, &encoding);
  offset += EncodeDiscards(*parent_game_, obs, offset, &encoding);
  offset += EncodeLastAction_(*parent_game_, obs, offset, &encoding);
  if (parent_game_->ObservationType() != HanabiGame::kMinimal) {
    offset += EncodeV0Belief_(*parent_game_, obs, offset, &encoding);
  }

  assert(offset == encoding.size());
  return encoding;
}

}  // namespace hanabi_learning_env
