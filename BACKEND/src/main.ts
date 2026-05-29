import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';
import { ValidationPipe } from '@nestjs/common';
import { getRepositoryToken } from '@nestjs/typeorm';
import { User } from './users/user.entity';
import * as bcrypt from 'bcrypt';

async function seedAdmin(app: any) {
  const repo = app.get(getRepositoryToken(User));
  const count = await repo.count();
  if (count === 0) {
    const passwordHash = await bcrypt.hash('Admin@123456', 12);
    const admin = repo.create({
      email: 'admin@pharmalink.local',
      username: 'admin',
      passwordHash,
      fullName: 'Administrator',
      role: 'admin',
      isActive: true,
    });
    await repo.save(admin);
    console.log('✅ Default admin seeded — username: admin / password: Admin@123456');
  }
}

async function bootstrap() {
  const app = await NestFactory.create(AppModule);

  app.enableCors({ origin: '*' });
  app.useGlobalPipes(new ValidationPipe({ transform: true }));
  app.setGlobalPrefix('api');

  await seedAdmin(app);

  const port = process.env.PORT || 3000;
  await app.listen(port);
  console.log(`PharmaLink Backend running on http://localhost:${port}/api`);
}
bootstrap();
