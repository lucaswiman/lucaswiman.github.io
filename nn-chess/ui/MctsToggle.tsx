// Shared "Use MCTS" checkbox + sims slider widget used by the three
// play pages (opponent, imitator, Maia). Page-specific extras (cPuct
// on the opponent page, temperature on the imitator page, calibration
// caption on the Maia page) stay inline at the call site — this
// component only owns what all three actually share.

interface MctsToggleProps {
  useMcts: boolean;
  onUseMctsChange: (v: boolean) => void;
  simulations: number;
  onSimulationsChange: (v: number) => void;
  minSims: number;
  maxSims: number;
  step?: number;
  /** Suffix shown after the numeric sims value, e.g. "sims/move". */
  simsSuffix?: string;
}

export default function MctsToggle({
  useMcts,
  onUseMctsChange,
  simulations,
  onSimulationsChange,
  minSims,
  maxSims,
  step = 1,
  simsSuffix,
}: MctsToggleProps) {
  const disabledColor = useMcts ? undefined : '#aaa';
  return (
    <>
      <label style={{ display: 'flex', gap: '6px', alignItems: 'center', cursor: 'pointer' }}>
        <input
          type="checkbox"
          checked={useMcts}
          onChange={e => onUseMctsChange(e.target.checked)}
        />
        <strong>Use MCTS</strong>
      </label>
      <label
        style={{
          display: 'flex',
          gap: '8px',
          alignItems: 'center',
          cursor: useMcts ? 'pointer' : 'not-allowed',
        }}
      >
        <span style={{ color: disabledColor }}>MCTS sims:</span>
        <input
          type="range"
          min={minSims}
          max={maxSims}
          step={step}
          value={simulations}
          onChange={e => onSimulationsChange(Number(e.target.value))}
          disabled={!useMcts}
          style={{ width: '120px', opacity: useMcts ? 1 : 0.4 }}
        />
        <span style={{ fontFamily: 'monospace', minWidth: '3em', color: disabledColor }}>
          {simulations}
          {simsSuffix ? ` ${simsSuffix}` : ''}
        </span>
      </label>
    </>
  );
}
