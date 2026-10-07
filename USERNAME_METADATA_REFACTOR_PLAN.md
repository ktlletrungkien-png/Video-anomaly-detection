# Kế hoạch refactor username từ SFU metadata

## 1. Mục tiêu

Chuyển hoàn toàn nguồn username/display name của participant sang metadata do `mezon-sfu` gửi trong `Member`.

Quy ước dữ liệu:

```text
Member.UserID   -> participant_identity
Member.Metadata -> "display_name;avatar_url"
```

Agent lấy phần trước dấu `;` làm username rồi gửi thẳng cùng `participant_identity` sang orchestrator. Orchestrator lưu trực tiếp, không gọi `agents-bot` để resolve username nữa.

Quy tắc persistence được chốt theo hướng **first observed username wins**:

- Username được lưu ở lần đầu participant xuất hiện qua `room_snapshot` hoặc `peer_joined`.
- Nếu `participant_identity` đã tồn tại trong room thì bỏ qua, không cập nhật username và không thay timestamp join.
- Không đồng bộ thay đổi display name giữa phiên từ `peer_updated`.
- Username rỗng vẫn được chấp nhận; UI fallback về identity và không có luồng lookup/fill lại từ service khác.

Luồng sau refactor:

```text
mezon-sfu
    |
    | Member { user_id, metadata }
    v
Go agent
    |
    | participant_identity + username
    v
orchestrator_service
    |
    | save participant / save batch participants
    v
Room.participants
```

`agents-bot` không còn tham gia lấy username participant. Service tiếp tục giữ luồng chat và bot profile cho các caller hiện có:

```text
Mezon ChannelMessage
    -> agents-bot
    -> orchestrator /agent_push_chat_external
    -> SSE chat_external
```

Không giữ fallback username qua agents-bot và không cần tương thích với agent cũ.

## 2. Phạm vi giữ lại và loại bỏ

### Giữ lại

- Agent đăng ký/hủy đăng ký room với agents-bot để agents-bot biết room nào đang active và được phép forward chat.
- Agents-bot nhận `ChannelMessage` và gọi `agent_push_chat_external`.
- Luồng push chat từ Go agent nếu vẫn phục vụ chat qua SFU.
- API bot profile cùng `accountAPI`, `BotProfile` và `GetBotProfile` vì vẫn có caller bên ngoài luồng refactor này.
- `participant_identity` dạng numeric Mezon user ID.

### Loại bỏ

- Orchestrator gọi agents-bot để resolve username.
- `AgentsBotUserClient` và cache username trong orchestrator.
- User resolver/cache trong agents-bot.
- Các API user/participant lookup của agents-bot.
- Username lấy từ `RoomMessage.Name` rồi gọi `/external/participant-chat`.
- `savedUsers`, `SaveExternalChatParticipant` và `force_save_participant` phục vụ luồng cũ.
- Config và tài liệu mô tả agents-bot là nguồn username.

## 3. Các bước thực hiện

### Bước 1. Hoàn thiện contract metadata ở signaling

Files:

- `agents/internal/signaling/messages.go`
- `agents/internal/signaling/messages_test.go`

Công việc:

1. Giữ field đã thêm trong `Member`:

   ```go
   Metadata string `json:"metadata"` // "display_name;avatar_url"
   ```

2. Thêm một helper parse metadata dùng chung.
3. Dùng dấu `;` đầu tiên để tách dữ liệu.
4. Trim khoảng trắng của display name.
5. Metadata không có dấu `;` được xem là chỉ chứa display name.
6. Không dùng display name cho authorization; identity vẫn lấy từ `UserID`.
7. Cập nhật payload test của:
   - `room_snapshot`
   - `peer_joined`
   - `peer_updated`
8. Thêm test cho Unicode, metadata rỗng và metadata không có avatar.

Kết quả cần đạt:

- `Member.Metadata` được decode và parse chính xác ở cả snapshot, joined và updated.

### Bước 2. Đổi request model của orchestrator sang participant object

File:

- `Architect_MultiClient_Server/orchestrator_service/models/room_registry_models.py`

Tạo model participant dùng chung:

```json
{
  "participant_identity": "123456",
  "username": "Nguyen Van A"
}
```

Thay đổi contract:

1. `ParticipantSnapshotRequest` bỏ `participant_identities`.
2. Thêm `participants: list[ParticipantRequest]`.
3. `ParticipantJoinedRequest` thêm field `username`.
4. Username lấy trực tiếp từ SFU nên không cần model hay field thể hiện trạng thái resolve/fallback.
5. Có thể giữ username optional/rỗng để room không fail nếu metadata rỗng, nhưng tuyệt đối không gọi agents-bot để bổ sung.

Snapshot mới:

```json
{
  "room_name": "123",
  "room_id": "00000000-0000-0000-0000-000000000000",
  "participants": [
    {
      "participant_identity": "456",
      "username": "Nguyen Van A"
    }
  ]
}
```

Joined mới:

```json
{
  "room_name": "123",
  "room_id": "00000000-0000-0000-0000-000000000000",
  "participant_identity": "456",
  "username": "Nguyen Van A"
}
```

Kết quả cần đạt:

- Orchestrator chỉ nhận participant object hoàn chỉnh từ agent.
- Contract cũ chỉ có identity bị loại bỏ.

### Bước 3. Thay toàn bộ đoạn resolve username trong orchestrator bằng lưu trực tiếp

File:

- `Architect_MultiClient_Server/orchestrator_service/api/v2/endpoints/room_registry_api.py`

Thay đổi `/participant/snapshot`:

1. Nhận `request.participants`.
2. Deduplicate theo `participant_identity` nếu cần.
3. Chuyển trực tiếp danh sách này sang `transcription_service.save_participants_batch`.
4. Không gọi `resolve_agents_bot_usernames`.
5. `resolved_username_count` được tính bằng số participant có username không rỗng.
6. `unresolved_username_count` là phần còn lại.

Thay đổi `/participant-joined`:

1. Lấy `request.participant_identity` và `request.username`.
2. Gọi trực tiếp `transcription_service.save_participant` hoặc hàm upsert participant.
3. Không gọi `resolve_agents_bot_usernames`.

Xóa ngay:

- Import `resolve_agents_bot_usernames`.
- Mọi nhánh fallback sang agents-bot.
- Log liên quan đến resolve username qua agents-bot.

Kết quả cần đạt:

- Snapshot và joined không tạo bất kỳ HTTP request nào sang agents-bot.

### Bước 4. Xóa kết nối username giữa orchestrator và agents-bot

Files:

- `Architect_MultiClient_Server/orchestrator_service/services/agents_bot_user_client.py`
- `Architect_MultiClient_Server/orchestrator_service/main.py`
- `Architect_MultiClient_Server/orchestrator_service/config/application_config.py`
- `Architect_MultiClient_Server/orchestrator_service/.env.example`
- `Architect_MultiClient_Server/orchestrator_service/README.md`

Công việc:

1. Xóa `services/agents_bot_user_client.py`.
2. Xóa import và call `close_agents_bot_user_client` trong lifecycle của orchestrator.
3. Xóa `AgentsBotConfig` khỏi application config.
4. Xóa `self.agents_bot` khỏi `Config`.
5. Xóa `AGENTS_BOT_BASE_URL` khỏi env/example và tài liệu của orchestrator.
6. Rà lại `httpx`; chỉ remove dependency nếu không còn module khác sử dụng.

Lưu ý:

- Không xóa `AGENTS_BOT_BASE_URL` ở Go agent/worker-manager vì agent vẫn cần register/unregister room với agents-bot cho push chat.

Kết quả cần đạt:

- Orchestrator không còn biết tới agents-bot ngoài việc nhận request push chat từ nó.

### Bước 5. Cập nhật Go orchestrator client để gửi username

File:

- `agents/internal/orchestratorclient/client.go`

Công việc:

1. Tạo DTO participant gồm:
   - `participant_identity`
   - `username`
2. Đổi `ParticipantSnapshot` nhận danh sách DTO participant.
3. Payload snapshot gửi field `participants` thay vì `participant_identities`.
4. Đổi `ParticipantJoined` nhận thêm username.
5. Cập nhật comment cũ nói rằng orchestrator tự resolve username qua agents-bot.
6. Thêm test HTTP payload bằng `httptest.Server` nếu package chưa có test.

Kết quả cần đạt:

- JSON do Go agent gửi khớp hoàn toàn với Pydantic model mới.

### Bước 6. Gửi identity và username từ `room_snapshot`

File:

- `agents/cmd/agent/main.go`

Tại `onRoomSnapshot`:

1. Tiếp tục `UpsertRoster` cho từng member.
2. Với mỗi member:
   - `participant_identity = strconv.FormatInt(member.UserID, 10)`.
   - `username = parseDisplayName(member.Metadata)`.
3. Tạo danh sách participant object.
4. Gọi `ParticipantSnapshot` với danh sách mới.
5. Không thực hiện lookup username ở bất kỳ service nào khác.

Kết quả cần đạt:

- Participant đã có mặt trước khi agent join được lưu cả identity và username trong một request batch.

### Bước 7. Gửi identity và username từ `peer_joined`

File:

- `agents/cmd/agent/main.go`

Tại `onPeerJoined`:

1. Parse `peer.Metadata`.
2. Gọi `ParticipantJoined` với identity + username.
3. Giữ timeout và goroutine hiện tại để không block signaling read loop.

Kết quả cần đạt:

- Participant join sau snapshot được lưu tên ngay, không cần agents-bot.

### Bước 8. Không persist username từ `peer_updated`

Quyết định:

1. `onPeerUpdated` chỉ tiếp tục cập nhật roster phục vụ RTC như hiện tại.
2. Không parse metadata để gọi orchestrator.
3. Không cập nhật username đã lưu khi participant đổi display name giữa phiên.
4. Username được ghi nhận lần đầu từ `room_snapshot` hoặc `peer_joined` và giữ nguyên trong suốt room.

Lý do:

- SFU đã cung cấp username ngay khi participant xuất hiện.
- Nếu SFU không có username ở thời điểm đó thì reconnect cũng không bổ sung được nguồn dữ liệu mới đáng tin cậy.
- Việc giữ tên đầu tiên giúp persistence đơn giản, ổn định timestamp và không phát sinh request khi peer chỉ đổi mute/role.

Kết quả cần đạt:

- `peer_updated` không tạo request participant persistence sang orchestrator.
- Đổi display name giữa phiên không thay đổi username đã lưu trong room.

### Bước 9. Chuẩn hóa persistence participant

Files:

- `Architect_MultiClient_Server/orchestrator_service/services/transcription_service.py`
- `Architect_MultiClient_Server/orchestrator_service/services/postgresql/pg_transcript_repository.py`

Quy tắc:

1. Identity chưa tồn tại: thêm participant cùng username và timestamp hiện tại.
2. Identity đã tồn tại: bỏ qua hoàn toàn, không append duplicate, không cập nhật username và không thay timestamp join.
3. Username đầu tiên được lưu là username được giữ cho toàn bộ vòng đời room.
4. Batch snapshot và single participant dùng cùng quy tắc identity duy nhất.
5. Snapshot deduplicate các entry trùng identity trong cùng request; repository tiếp tục chống race giữa snapshot và `peer_joined`.
6. Username rỗng không làm lỗi room; UI vẫn fallback về identity như hiện tại và không có luồng fill lại từ agents-bot.

Không cần thay schema database vì `Room.participants` đang là JSONB.

Kết quả cần đạt:

- Một room chỉ có một entry cho mỗi `participant_identity`.
- Snapshot và joined có thể chạy gần nhau mà không tạo duplicate.
- Username và timestamp của entry đầu tiên không bị ghi đè bởi các event sau.

Code hiện tại đã dùng `save_participant` theo kiểu insert-once và `save_batch_participants_atomic` để chống race. Không cần bổ sung logic update username ở bước này.

### Bước 10. Xóa luồng lấy username từ external room message

Agent files:

- `agents/cmd/agent/main.go`
- `agents/internal/orchestratorclient/client.go`

Orchestrator files:

- `Architect_MultiClient_Server/orchestrator_service/models/room_registry_models.py`
- `Architect_MultiClient_Server/orchestrator_service/api/v2/endpoints/room_registry_api.py`
- `Architect_MultiClient_Server/orchestrator_service/services/transcription_service.py`
- `Architect_MultiClient_Server/orchestrator_service/services/postgresql/pg_transcript_repository.py`

Xóa:

1. `session.savedUsers`.
2. Block lấy `message.Name` và lưu participant trong `onRoomMessage`.
3. `SaveExternalChatParticipant`.
4. `ParticipantChatRequest`.
5. Endpoint `/external/participant-chat`.
6. `TranscriptionService.force_save_participant`.
7. `PgTranscriptRepository.force_save_participant` nếu không còn caller.

Giữ lại:

- `PushChatExternal` và phần forward nội dung chat.

Kết quả cần đạt:

- `RoomMessage.Name` không còn là nguồn username participant.
- Chat và participant persistence là hai trách nhiệm tách biệt.

### Bước 11. Thu gọn agents-bot, giữ push chat và bot profile

Files chính:

- `agents-bot/internal/gateway/gateway.go`
- `agents-bot/internal/userresolver/`
- `agents/internal/agentsbotclient/client.go`

Giữ lại:

1. Mezon client login/lifecycle.
2. `ChannelMessage` listener.
3. `forwardChatIfActive`.
4. Active room map.
5. `POST /api/rooms/register`.
6. `POST /api/rooms/unregister`.
7. Health endpoint tối giản.
8. Orchestrator client dùng để push chat.
9. `GET /api/bot/profile` và các thành phần hỗ trợ gồm `accountAPI`, `BotProfile`, `GetBotProfile`.

Xóa:

1. Package `internal/userresolver`.
2. Field `resolver` trong `Gateway`.
3. `VoiceJoinedEvent` handler.
4. `VoiceLeavedEvent` handler.
5. Cache user từ `ChannelMessage`.
6. `GET /api/users/{id}`.
7. `POST /api/users`.
8. `GET /api/rooms/{room_name}/participants`.
9. `user_cache` khỏi health response.
10. Các DTO/helper user response và clan context không còn dùng.

Kết quả cần đạt:

- Agents-bot không còn lưu hoặc trả username participant.
- Agents-bot down chỉ ảnh hưởng external chat forwarding, không ảnh hưởng participant name.
- Bot profile tiếp tục hoạt động cho caller hiện có và nằm ngoài luồng username participant.

### Bước 12. Cập nhật config và tài liệu

Rà và cập nhật:

1. `agents/README.md`.
2. `agents-bot/README.md`.
3. `Architect_MultiClient_Server/orchestrator_service/README.md`.
4. Các `.env.example`.
5. Package comments trong gateway và orchestrator client.

Trạng thái config cuối cùng:

- Go agent giữ `AGENTS_BOT_BASE_URL` để register/unregister room.
- Orchestrator không còn `AGENTS_BOT_BASE_URL`.
- Agents-bot giữ `ORCHESTRATOR_BASE_URL` và `INTERNAL_API_SECRET` để push chat.

### Bước 13. Chạy test và kiểm tra end-to-end

Go checks từ `agents/`:

```text
go test ./internal/signaling/...
go test ./internal/orchestratorclient/...
go test ./internal/rtcagent/...
go test ./cmd/agent/...
go test ./...
```

Python checks từ orchestrator:

```text
python -m compileall orchestrator_service
pytest
```

E2E checklist:

1. [ ] Agent join room có participant sẵn: snapshot lưu identity + username.
2. [ ] Participant join sau: joined event lưu identity + username.
3. [ ] Participant đổi display name giữa phiên: DB giữ username đầu tiên và không phát sinh request persistence từ `peer_updated`.
4. [ ] Metadata Unicode hiển thị đúng.
5. [ ] Metadata rỗng không làm room fail; UI fallback identity.
6. [ ] Reconnect agent không tạo participant duplicate và không ghi đè username/timestamp đã lưu.
7. [ ] Participant list trả username đã lưu.
8. [ ] Summary sử dụng username thay vì `user-N` khi metadata có tên.
9. [ ] Internal/SFU chat vẫn được forward.
10. [ ] External Mezon chat qua agents-bot vẫn được forward.
11. [ ] Orchestrator không gọi bất kỳ user lookup API nào của agents-bot.
12. [ ] Agents-bot không còn user cache hoặc participant lookup endpoint.

## 4. Thứ tự commit khuyến nghị

1. `refactor(orchestrator): accept and persist participant username from agent`
2. `refactor(agent): forward SFU member metadata with participant events`
3. `refactor: remove external chat participant persistence flow`
4. `refactor(orchestrator): remove agents-bot username lookup client`
5. `refactor(agents-bot): remove user resolver, keep chat and bot profile`
6. `docs: update participant username flow`

Các commit có thể được phát triển trong cùng branch. Khi deploy, orchestrator và agent phải được phát hành cùng một đợt vì contract snapshot thay đổi dứt khoát và không giữ tương thích payload cũ.

## 5. Definition of Done

1. [ ] `Member.Metadata` là nguồn duy nhất của username participant.
2. [ ] Snapshot và joined gửi identity + username từ agent; updated không persist username.
3. [ ] Orchestrator lưu trực tiếp dữ liệu agent gửi.
4. [ ] Orchestrator không còn client/config kết nối agents-bot để resolve username.
5. [ ] Không còn `/external/participant-chat` và `RoomMessage.Name` persistence.
6. [ ] Agents-bot giữ room registration, push chat và bot profile; không tham gia resolve username participant.
7. [ ] Không còn `userresolver` hoặc user lookup API trong agents-bot.
8. [ ] Participant persistence không tạo duplicate; username và timestamp đầu tiên không bị ghi đè.
9. [ ] Participant list và summary sử dụng username đã lưu.
10. [ ] Toàn bộ unit/integration/E2E checks pass.
