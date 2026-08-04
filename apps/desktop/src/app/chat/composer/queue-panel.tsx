import { StatusRow } from '@/components/chat/status-row'
import { StatusSection } from '@/components/chat/status-section'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Tip } from '@/components/ui/tooltip'
import { type Translations, useI18n } from '@/i18n'
import { CornerDownLeft, iconSize, Pencil, RefreshCw, Trash2 } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { isQueuedPromptStuck, type QueuedPromptEntry } from '@/store/composer-queue'

interface QueuePanelProps {
  busy: boolean
  editingId: null | string
  entries: QueuedPromptEntry[]
  onDelete: (id: string) => void
  onEdit: (entry: QueuedPromptEntry) => void
  /** Lift a park (explicit Stop/Esc halt) and let the queue flow again. */
  onResume: () => void
  /** Give a dead-lettered entry a fresh auto-drain budget. */
  onRetry: (id: string) => void
  onSendNow: (id: string) => void
  /** True after an explicit halt: entries wait until resumed / sent / edited. */
  parked: boolean
}

const entryPreview = (entry: QueuedPromptEntry, c: Translations['composer']) =>
  (entry.displayText ?? entry.text).trim() || (entry.attachments.length > 0 ? c.attachmentOnly : c.emptyTurn)

/** Why this entry stopped trying, in the user's words. The stored `lastError`
 *  (a full attachment path, or the gateway's own message) is appended verbatim
 *  and never truncated — a half path is what made this class of failure
 *  impossible to diagnose in the first place. */
const stuckDetail = (entry: QueuedPromptEntry, c: Translations['composer']): string => {
  const headline = c.queueStuckReason[entry.stuckReason ?? 'drain-failed']

  return entry.lastError ? `${headline} — ${entry.lastError}` : headline
}

export function QueuePanel({
  busy,
  editingId,
  entries,
  onDelete,
  onEdit,
  onResume,
  onRetry,
  onSendNow,
  parked
}: QueuePanelProps) {
  const { t } = useI18n()
  const c = t.composer

  if (entries.length === 0) {
    return null
  }

  // A dead-lettered entry is a failure the user has to act on, so it must not
  // hide behind a collapsed disclosure that only says "N Queued". Open the
  // section when something needs attention; a healthy queue stays tucked away.
  const hasStuck = entries.some(isQueuedPromptStuck)

  return (
    // Keyed on the park flag AND on whether anything is dead-lettered:
    // StatusSection owns its collapse state from defaultCollapsed, so it has to
    // remount when either flips. A Stop must EXPAND the panel — the halted
    // prompts' only presence is here, and leaving them behind a collapsed
    // "N queued" pill is how they read as vanished. Same for a stuck entry: it
    // is a failure the user has to act on, not something to hide behind a pill.
    <StatusSection
      accessory={
        parked ? (
          <Tip label={c.queueResumeTip}>
            <Button
              className="text-muted-foreground/75 hover:text-foreground/90"
              onClick={onResume}
              size="micro"
              type="button"
              variant="text"
            >
              {c.queueResume}
            </Button>
          </Tip>
        ) : undefined
      }
      defaultCollapsed={!parked && !hasStuck}
      icon={<Codicon className="text-muted-foreground/70" name={parked ? 'debug-pause' : 'layers'} size="0.8rem" />}
      key={`${parked ? 'parked' : 'flowing'}-${hasStuck ? 'stuck' : 'ok'}`}
      label={parked ? c.queuedPaused(entries.length) : c.queued(entries.length)}
    >
      {entries.map(entry => {
        const isEditing = editingId === entry.id
        const attachmentsCount = entry.attachments.length
        // A dead-lettered entry has stopped retrying on its own. It stays
        // visible with its reason and its two ways out — Retry (fresh attempt
        // budget) and Delete — because silently dropping a turn the user wrote
        // is worse than showing them a failure they can act on.
        const isStuck = isQueuedPromptStuck(entry)

        return (
          <StatusRow
            className={cn(
              'border border-transparent',
              isEditing && 'border-[color-mix(in_srgb,var(--dt-composer-ring)_40%,transparent)] bg-accent/25',
              isStuck && 'border-destructive/30 bg-destructive/5'
            )}
            key={entry.id}
            trailing={
              <>
                <Tip label={c.queueEdit}>
                  <Button
                    aria-label={c.queueEdit}
                    className="size-5 rounded-md"
                    disabled={Boolean(editingId) && !isEditing}
                    onClick={() => onEdit(entry)}
                    size="icon-xs"
                    type="button"
                    variant="ghost"
                  >
                    <Pencil className={iconSize.xs} />
                  </Button>
                </Tip>
                {isStuck ? (
                  <Tip label={c.queueRetry}>
                    <Button
                      aria-label={c.queueRetry}
                      className="size-5 rounded-md"
                      disabled={isEditing}
                      onClick={() => onRetry(entry.id)}
                      size="icon-xs"
                      type="button"
                      variant="ghost"
                    >
                      <RefreshCw className={iconSize.xs} />
                    </Button>
                  </Tip>
                ) : (
                  <Tip label={busy ? c.queueSendNext : c.queueSend}>
                    <Button
                      aria-label={busy ? c.queueSendNext : c.queueSend}
                      className="size-5 rounded-md"
                      disabled={isEditing}
                      onClick={() => onSendNow(entry.id)}
                      size="icon-xs"
                      type="button"
                      variant="ghost"
                    >
                      <CornerDownLeft className={iconSize.xs} />
                    </Button>
                  </Tip>
                )}
                <Tip label={c.queueDelete}>
                  <Button
                    aria-label={c.queueDelete}
                    className="size-5 rounded-md"
                    onClick={() => onDelete(entry.id)}
                    size="icon-xs"
                    type="button"
                    variant="ghost"
                  >
                    <Trash2 className={iconSize.xs} />
                  </Button>
                </Tip>
              </>
            }
            trailingVisible={isEditing || isStuck}
          >
            <div className="min-w-0 flex-1">
              <p
                className={cn(
                  'truncate text-[0.73rem] leading-4 text-foreground/92',
                  isStuck && 'text-foreground/70 line-through decoration-foreground/25'
                )}
              >
                {entryPreview(entry, c)}
              </p>
              {isStuck && (
                <p
                  className="mt-0.5 break-all text-[0.64rem] leading-4 text-destructive/90"
                  title={stuckDetail(entry, c)}
                >
                  {stuckDetail(entry, c)}
                </p>
              )}
              {(attachmentsCount > 0 || isEditing) && (
                <div className="mt-0.5 flex items-center gap-1.5 text-[0.64rem] text-muted-foreground/75">
                  {attachmentsCount > 0 && <span>{c.attachments(attachmentsCount)}</span>}
                  {isEditing && (
                    <span className="text-[color-mix(in_srgb,var(--dt-composer-ring)_78%,var(--muted-foreground))]">
                      {c.editingInComposer}
                    </span>
                  )}
                </div>
              )}
            </div>
          </StatusRow>
        )
      })}
    </StatusSection>
  )
}
