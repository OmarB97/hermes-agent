import { useCallback, useEffect, useState } from 'react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { useI18n } from '@/i18n'
import {
  acceptCloudChannelInvite,
  type CloudChannel,
  loadCloudChannels
} from '@/lib/cloud-share'
import { cn } from '@/lib/utils'

interface CloudChannelsDialogProps {
  onOpenChange: (open: boolean) => void
  open: boolean
}

const channelLabel = (channel: CloudChannel) => {
  const title = channel.title?.trim()

  return title || channel.id
}

const formatSeq = (seq: null | number | undefined) => (typeof seq === 'number' ? String(seq) : '0')

export function CloudChannelsDialog({ onOpenChange, open }: CloudChannelsDialogProps) {
  const { t } = useI18n()
  const s = t.sidebar
  const [channels, setChannels] = useState<CloudChannel[]>([])
  const [inviteToken, setInviteToken] = useState('')
  const [loading, setLoading] = useState(false)
  const [submitting, setSubmitting] = useState(false)

  const refresh = useCallback(async () => {
    setLoading(true)

    try {
      const result = await loadCloudChannels()

      if (result) {
        setChannels(result)
      }
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    if (open) {
      void refresh()
    }
  }, [open, refresh])

  const acceptInvite = useCallback(async () => {
    setSubmitting(true)

    try {
      const result = await acceptCloudChannelInvite(inviteToken)

      if (result) {
        setInviteToken('')
        await refresh()
      }
    } finally {
      setSubmitting(false)
    }
  }, [inviteToken, refresh])

  return (
    <Dialog onOpenChange={onOpenChange} open={open}>
      <DialogContent className="max-w-lg gap-4">
        <DialogHeader>
          <DialogTitle>{s.cloudChannelsTitle}</DialogTitle>
          <DialogDescription>{s.cloudChannelsDesc}</DialogDescription>
        </DialogHeader>

        <div className="grid gap-2">
          <div className="flex min-w-0 gap-2">
            <Input
              aria-label={s.cloudInviteTokenAria}
              className="min-w-0 flex-1"
              onChange={event => setInviteToken(event.currentTarget.value)}
              onKeyDown={event => {
                if (event.key === 'Enter') {
                  event.preventDefault()
                  void acceptInvite()
                }
              }}
              placeholder={s.cloudInviteTokenPlaceholder}
              value={inviteToken}
            />
            <Button disabled={submitting || !inviteToken.trim()} onClick={() => void acceptInvite()} type="button">
              {submitting ? s.cloudInviteAccepting : s.cloudInviteAccept}
            </Button>
          </div>

          <div className="flex min-h-7 items-center justify-between gap-2">
            <div className="text-xs font-medium text-(--ui-text-secondary)">{s.cloudChannelsListTitle}</div>
            <Button disabled={loading} onClick={() => void refresh()} size="sm" type="button" variant="ghost">
              <Codicon name="refresh" />
              {loading ? s.cloudChannelsRefreshing : s.cloudChannelsRefresh}
            </Button>
          </div>

          <div
            className={cn(
              'max-h-72 overflow-y-auto rounded-md border border-(--ui-stroke-tertiary) bg-(--ui-control-background)',
              channels.length === 0 && 'p-3'
            )}
          >
            {channels.length === 0 ? (
              <div className="text-xs text-(--ui-text-tertiary)">
                {loading ? s.cloudChannelsLoading : s.cloudChannelsEmpty}
              </div>
            ) : (
              <div className="grid gap-px p-1">
                {channels.map(channel => (
                  <div
                    className="grid min-h-12 grid-cols-[minmax(0,1fr)_auto] gap-2 rounded-[4px] px-2 py-1.5 hover:bg-(--ui-control-hover-background)"
                    key={channel.id}
                  >
                    <div className="min-w-0">
                      <div className="truncate text-xs font-medium text-(--ui-text-primary)">
                        {channelLabel(channel)}
                      </div>
                      <div className="mt-0.5 truncate text-[0.6875rem] text-(--ui-text-tertiary)">
                        {channel.id}
                      </div>
                    </div>
                    <div className="flex shrink-0 flex-col items-end justify-center gap-1">
                      <Badge variant={channel.is_owner ? 'default' : 'muted'}>
                        {channel.is_owner ? s.cloudChannelsOwner : channel.your_permission || s.cloudChannelsRead}
                      </Badge>
                      <span className="text-[0.6875rem] text-(--ui-text-tertiary)">
                        {s.cloudChannelsSeq(formatSeq(channel.last_seq))}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        <DialogFooter>
          <Button onClick={() => onOpenChange(false)} type="button" variant="ghost">
            {s.cloudChannelsClose}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
