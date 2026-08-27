CREATE TABLE `users` (
	`id` int AUTO_INCREMENT NOT NULL,
	`openId` varchar(64) NOT NULL,
	`name` text,
	`email` varchar(320),
	`loginMethod` varchar(64),
	`role` enum('user','admin') NOT NULL DEFAULT 'user',
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	`updatedAt` timestamp NOT NULL DEFAULT (now()) ON UPDATE CURRENT_TIMESTAMP,
	`lastSignedIn` timestamp NOT NULL DEFAULT (now()),
	CONSTRAINT `users_id` PRIMARY KEY(`id`),
	CONSTRAINT `users_openId_unique` UNIQUE(`openId`)
);
--> statement-breakpoint
CREATE TABLE `video_jobs` (
	`id` varchar(36) NOT NULL,
	`ownerId` int NOT NULL,
	`workerId` varchar(36) NOT NULL,
	`sourceKey` varchar(256) NOT NULL,
	`sourceName` varchar(256) NOT NULL,
	`status` enum('queued','claimed','processing','awaiting_review','complete','failed','cancelled') NOT NULL DEFAULT 'queued',
	`stage` varchar(64) NOT NULL DEFAULT 'queued',
	`progress` int NOT NULL DEFAULT 0,
	`configJson` json NOT NULL,
	`outputLocalPath` text,
	`errorMessage` text,
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	`updatedAt` timestamp NOT NULL DEFAULT (now()) ON UPDATE CURRENT_TIMESTAMP,
	`startedAt` timestamp,
	`completedAt` timestamp,
	CONSTRAINT `video_jobs_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
CREATE TABLE `video_segments` (
	`id` varchar(36) NOT NULL,
	`jobId` varchar(36) NOT NULL,
	`position` int NOT NULL,
	`startMs` int NOT NULL,
	`endMs` int NOT NULL,
	`asrTextZh` text,
	`ocrTextZh` text,
	`sourceTextZh` text,
	`translatedTextVi` text,
	`voicePath` text,
	`voiceDurationMs` int,
	`confidence` int NOT NULL DEFAULT 0,
	`needsReview` int NOT NULL DEFAULT 0,
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	`updatedAt` timestamp NOT NULL DEFAULT (now()) ON UPDATE CURRENT_TIMESTAMP,
	CONSTRAINT `video_segments_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
CREATE TABLE `worker_clients` (
	`id` varchar(36) NOT NULL,
	`ownerId` int NOT NULL,
	`displayName` varchar(96) NOT NULL,
	`tokenHash` varchar(64) NOT NULL,
	`inventoryJson` json NOT NULL,
	`lastSeenAt` timestamp,
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	`updatedAt` timestamp NOT NULL DEFAULT (now()) ON UPDATE CURRENT_TIMESTAMP,
	CONSTRAINT `worker_clients_id` PRIMARY KEY(`id`),
	CONSTRAINT `worker_clients_tokenHash_unique` UNIQUE(`tokenHash`)
);
--> statement-breakpoint
ALTER TABLE `video_jobs` ADD CONSTRAINT `video_jobs_ownerId_users_id_fk` FOREIGN KEY (`ownerId`) REFERENCES `users`(`id`) ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE `video_jobs` ADD CONSTRAINT `video_jobs_workerId_worker_clients_id_fk` FOREIGN KEY (`workerId`) REFERENCES `worker_clients`(`id`) ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE `video_segments` ADD CONSTRAINT `video_segments_jobId_video_jobs_id_fk` FOREIGN KEY (`jobId`) REFERENCES `video_jobs`(`id`) ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE `worker_clients` ADD CONSTRAINT `worker_clients_ownerId_users_id_fk` FOREIGN KEY (`ownerId`) REFERENCES `users`(`id`) ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
CREATE INDEX `job_worker_status_idx` ON `video_jobs` (`workerId`,`status`);--> statement-breakpoint
CREATE INDEX `job_owner_created_idx` ON `video_jobs` (`ownerId`,`createdAt`);--> statement-breakpoint
CREATE INDEX `segment_job_position_idx` ON `video_segments` (`jobId`,`position`);--> statement-breakpoint
CREATE INDEX `worker_owner_idx` ON `worker_clients` (`ownerId`);