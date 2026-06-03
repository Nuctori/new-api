/*
Copyright (C) 2023-2026 QuantumNous

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

For commercial licensing, please contact support@quantumnous.com
*/
import assert from 'node:assert/strict'
import { describe, test } from 'node:test'
import {
  CHANNEL_FORM_DEFAULT_VALUES,
  transformChannelToFormDefaults,
  transformFormDataToCreatePayload,
  transformFormDataToUpdatePayload,
} from './channel-form'
import type { Channel } from '../types'

function createChannel(overrides: Partial<Channel> = {}): Channel {
  return {
    id: 1,
    type: 43,
    key: '',
    openai_organization: null,
    test_model: null,
    status: 1,
    name: 'deepseek',
    weight: 0,
    created_time: 0,
    test_time: 0,
    response_time: 0,
    base_url: 'https://api.deepseek.com',
    other: '',
    balance: 0,
    balance_updated_time: 0,
    models: 'deepseek-chat',
    group: 'default',
    used_quota: 0,
    model_mapping: null,
    status_code_mapping: null,
    priority: 0,
    auto_ban: 1,
    other_info: '',
    tag: null,
    setting: null,
    param_override: null,
    header_override: null,
    remark: '',
    max_input_tokens: 0,
    channel_info: {
      is_multi_key: false,
      multi_key_size: 0,
      multi_key_polling_index: 0,
      multi_key_mode: 'random',
    },
    settings: '{}',
    ...overrides,
  }
}

describe('channel responses compat form behavior', () => {
  test('defaults responses compatibility to disabled', () => {
    assert.equal(CHANNEL_FORM_DEFAULT_VALUES.responses_compat_mode, false)
  })

  test('loads explicit responses compatibility toggle from channel settings', () => {
    const defaults = transformChannelToFormDefaults(
      createChannel({
        settings: JSON.stringify({
          responses_compat_mode: true,
        }),
      })
    )

    assert.equal(defaults.responses_compat_mode, true)
  })

  test('create payload keeps explicit false for OpenAI-compatible channels', () => {
    const payload = transformFormDataToCreatePayload({
      ...CHANNEL_FORM_DEFAULT_VALUES,
      type: 43,
      name: 'ds',
      key: 'sk-test',
      models: 'deepseek-chat',
      group: ['default'],
      settings: '{"custom":"value"}',
      responses_compat_mode: false,
    })

    assert.deepEqual(JSON.parse(payload.channel.settings || '{}'), {
      custom: 'value',
      responses_compat_mode: false,
      upstream_model_update_check_enabled: false,
      upstream_model_update_auto_sync_enabled: false,
      upstream_model_update_ignored_models: [],
      upstream_model_update_last_detected_models: [],
      upstream_model_update_last_check_time: 0,
    })
  })

  test('update payload removes responses compatibility from unsupported channel types', () => {
    const payload = transformFormDataToUpdatePayload(
      {
        ...CHANNEL_FORM_DEFAULT_VALUES,
        type: 14,
        name: 'claude',
        models: 'claude-3-7-sonnet',
        group: ['default'],
        settings: '{"responses_compat_mode":true,"custom":"value"}',
        responses_compat_mode: true,
      },
      7
    )

    assert.deepEqual(JSON.parse(payload.settings || '{}'), {
      custom: 'value',
      allow_service_tier: false,
      allow_inference_geo: false,
      allow_speed: false,
      claude_beta_query: false,
      upstream_model_update_check_enabled: false,
      upstream_model_update_auto_sync_enabled: false,
      upstream_model_update_ignored_models: [],
      upstream_model_update_last_detected_models: [],
      upstream_model_update_last_check_time: 0,
    })
  })
})
