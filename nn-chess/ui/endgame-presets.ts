import type { Color } from '../core/rules/index.js';

export interface EndgamePreset {
  id: string;
  name: string;
  description: string;
  fen: string;
  winningSide: Color;
}

export const ENDGAME_PRESETS: readonly EndgamePreset[] = [
  {
    id: 'ladder-mate',
    name: 'Ladder mate (K+2R vs K)',
    description:
      'The easiest forced mate in chess — drive the king to the edge with the rooks. Good for teaching the policy head that mate exists.',
    fen: '4k3/8/8/8/8/8/8/R3K2R w - - 0 1',
    winningSide: 'w',
  },
  {
    id: 'kq-vs-k',
    name: 'Queen mate (K+Q vs K)',
    description:
      'Fundamental endgame. Box the king in with the queen, bring your king up, deliver mate on the edge.',
    fen: '4k3/8/8/8/8/8/8/4K2Q w - - 0 1',
    winningSide: 'w',
  },
  {
    id: 'kr-vs-k',
    name: 'Rook mate (K+R vs K)',
    description:
      'Classic technique: cut the king off, oppose with your king, walk the rook for the back-rank mate.',
    fen: '4k3/8/8/8/8/8/8/R3K3 w - - 0 1',
    winningSide: 'w',
  },
  {
    id: 'kp-vs-k-won',
    name: 'King + pawn vs king (won)',
    description:
      'White has the opposition and the key squares. Push the pawn to promotion. Short, forcing, value-head friendly.',
    fen: '8/8/8/3k4/8/3K4/3P4/8 w - - 0 1',
    winningSide: 'w',
  },
];

export const ENDGAME_PRESETS_BY_ID: ReadonlyMap<string, EndgamePreset> = new Map(
  ENDGAME_PRESETS.map(p => [p.id, p] as const),
);
