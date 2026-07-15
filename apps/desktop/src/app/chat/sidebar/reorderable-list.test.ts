import { describe, expect, it } from 'vitest'

import { sortableRowStyle } from './reorderable-list'

describe('sortableRowStyle', () => {
  it('keeps only the lifted row on the direct pointer transform when Motion owns the preview', () => {
    expect(
      sortableRowStyle({
        isDragging: true,
        previewOwnsLayout: true,
        transform: { y: 27 },
        transition: 'transform 200ms'
      })
    ).toEqual({
      transform: 'translate3d(0px, 27px, 0)',
      transition: undefined,
      willChange: 'transform'
    })

    expect(
      sortableRowStyle({
        isDragging: false,
        previewOwnsLayout: true,
        transform: { y: -24 },
        transition: 'transform 200ms'
      })
    ).toEqual({ transform: undefined, transition: undefined, willChange: undefined })
  })

  it('retains dnd-kit sibling transforms for standalone reorder lists', () => {
    expect(
      sortableRowStyle({
        isDragging: false,
        transform: { y: -24 },
        transition: 'transform 200ms'
      })
    ).toEqual({
      transform: 'translate3d(0px, -24px, 0)',
      transition: 'transform 200ms',
      willChange: undefined
    })
  })
})
